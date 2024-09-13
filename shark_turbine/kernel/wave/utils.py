from ..compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
)
from typing import Callable, Any, List, Tuple
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexExpr, IndexingContext, IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from ..ops.wave_ops import get_custom, Output, Write, MMA
from .constraints import HardwareConstraint, TilingConstraint
import torch.fx as fx
import shark_turbine.kernel.lang as tkl

from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)

import sympy
import torch
from iree.compiler import compile_str
import iree.runtime as rt
import iree.runtime.benchmark as bench

# TODO: Monkey-patching f16 support, need to fix in iree.
import numpy

bench.DTYPE_TO_ABI_TYPE[numpy.dtype(numpy.float16)] = "f16"


def canonicalize_module(module: Operation):
    with module.context, Location.unknown():
        transform_module = builtin_d.Module.create()
        transform_module_op = module.operation
        transform_module_op.attributes["transform.with_named_sequence"] = UnitAttr.get()
        with InsertionPoint(transform_module.body):
            named_sequence = transform_d.NamedSequenceOp(
                "__transform_main", [any_op_t()], []
            )
            with InsertionPoint(named_sequence.body):
                target = named_sequence.body.arguments[0]
                apply_patterns = transform_d.ApplyPatternsOp(target)
                with InsertionPoint(apply_patterns.regions[0].blocks[0]):
                    transform_d.apply_patterns_canonicalization()
                transform_d.apply_cse(target)
                transform_d.YieldOp([target])
        transform_interpreter.apply_named_sequence(
            module,
            transform_module.body.operations[0],
            transform_module,
        )


def run_test(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    func()
    # Print a separator between tests
    print("-----")
    return func


def print_trace(trace: CapturedTrace, custom_print: bool = True):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for subgraph in reversed(list(trace.region_graph.subgraphs.values())):
        print(subgraph)
        if custom_print:
            for node in subgraph.nodes:
                print(get_custom(node))


def DCE(trace: CapturedTrace):
    """
    Removes all operators that are not used in the graph,
    excluding output and global write nodes.
    Repeats this process till no more operators can be removed.
    """

    def is_removable_operator(node: fx.Node) -> bool:
        custom = get_custom(node)
        idxc = IndexingContext.current()
        is_global_write = isinstance(custom, Write) and (
            custom.type.address_space.subs(idxc.subs) == GLOBAL_ADDRESS_SPACE
            or custom.type.address_space.subs(idxc.subs)
            == tkl.AddressSpace.GLOBAL_MEMORY.value
        )

        return (
            not custom.users and not isinstance(custom, Output) and not is_global_write
        )

    while removable_nodes := trace.walk(is_removable_operator):
        for node in removable_nodes:
            get_custom(node).graph.erase_node(node)


def delinearize_index(index: IndexExpr, shape: list[int]) -> list[IndexExpr]:
    """
    Delinearizes a 1D index into a multi-dimensional index
    based on the shapes provided. The returned array contains
    the multi-dimensional index.

    Assume the index is x and the shape is [5, 4, 3]. In this case,
    this function returns [x % 3, (x // 3) % 4, (x // 12) % 5].

    """
    nd_index = []
    product = 1
    for i, size in enumerate(reversed(shape)):
        if i == 0:
            nd_index.append(index % size)
        else:
            nd_index.append(sympy.floor(index / product) % size)
        product *= size
    return nd_index[::-1]


def simplify_index(index: IndexExpr) -> IndexExpr:
    """
    Simplifies the index by applying the following bindings:
        - MMA acc_index bindings so the index of the MMA node is the acc_index.
    """
    mapping = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
    return subs_idxc(index.subs(mapping))


def get_mma_dimensional_mapping(trace: CapturedTrace) -> dict[IndexSymbol, int]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2).
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMA)

    mapping: dict[IndexSymbol, int] = {}
    for node in trace.walk(is_mma):
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape
        k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        mapping[m] = 0
        mapping[n] = 1
        mapping[k] = 2

    return mapping


def get_hardware_vector_size(
    dim: IndexSymbol,
    hardware_constraint: HardwareConstraint,
    mma_indices: dict[IndexSymbol, int],
) -> dict[IndexSymbol, int]:
    """
    Given a hardware constraint, return the vector sizes for the given dimension.
    This could be a hardware specific vector size or a user specified vector size.
    """
    if mma_indices:
        vector_size = hardware_constraint.mma_matrix_shapes[mma_indices[dim]]
    else:
        vector_size = hardware_constraint.vector_shapes[dim]
    return vector_size


def remove_global_indexing(
    index: dict[IndexSymbol, IndexSequence], tilingConstraints: list[TilingConstraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    workgroup and induction level indexing. This is necessary for writes to shared memory
    that operate on promoted memory.
    """
    workgroup_ids = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
    new_index = {key: index[key].subs({w: 0 for w in workgroup_ids}) for key in index}
    for key in new_index:
        for constraint in tilingConstraints:
            new_index[key] = new_index[key].subs({constraint.induction_var: 0})
    return new_index


def _invoke(vm_context, device, entry_function, inputs, outputs):
    arg_list = rt.VmVariantList(len(inputs))
    ret_list = rt.VmVariantList(len(outputs))

    for input in inputs:
        input_cpu = input.cpu().contiguous()
        device_array = rt.asdevicearray(device, input_cpu)
        arg_list.push_ref(device_array._buffer_view)

    vm_context.invoke(entry_function, arg_list, ret_list)

    for i, ret in enumerate(outputs):
        device_buffer_view = rt.HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
        device_array = rt.DeviceArray(device, device_buffer_view)

        # TODO: Make to_host accept out array/buffer, so we can avoid extra data copy.
        host_array = device_array.to_host()

        # Convert to torch tensor without actually importing torch.
        ret[:] = type(ret)(host_array)


def _write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


def _print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        _write_file(filename, "w", res)
    else:
        print(res)


def compile_and_invoke(
    asm: str,
    func_name: str,
    config: dict[str, str],
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    run: bool = False,
    run_bench: bool = False,
):
    backend = config["backend"]
    device = config["device"]
    flags = [
        f"--iree-hal-target-backends={backend}",
        "--iree-vm-bytecode-module-strip-source-map=true",
        "--iree-opt-strip-assertions=true",
        "--iree-vm-target-truncate-unsupported-floats",
    ]

    # TODO: More targets/backends support.
    if backend == "rocm":
        target = config["target"]
        flags.append(f"--iree-hip-target={target}")

    if config.get("print_ir_after_all", False):
        flags.append("--mlir-print-ir-after-all")

    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        bench_repetitions = config.get("benchmark_repetitions", None)
        bench_file = config.get("benchmark_results_file", None)

        benchmark_flags = {}

        if bench_batch_size is not None:
            flags.append(
                f"--iree-hal-benchmark-dispatch-repeat-count={bench_batch_size}"
            )
            benchmark_flags["batch_size"] = int(bench_batch_size)

        if bench_repetitions is not None:
            benchmark_flags["benchmark_repetitions"] = int(bench_repetitions)

    res = compile_str(asm, target_backends=[backend], extra_args=flags)

    dump_vmfb_file = config.get("dump_vmfb_file", None)
    if dump_vmfb_file is not None:
        _write_file(dump_vmfb_file, "wb", res)

    rt_config = rt.Config(device)
    device = rt_config.device
    vm_instance = rt_config.vm_instance
    mod = rt.VmModule.copy_buffer(vm_instance, res)

    vm_modules = [
        mod,
        rt.create_hal_module(vm_instance, device),
    ]
    ctx = rt.SystemContext(
        vm_modules=vm_modules,
        config=rt_config,
    )

    if run:
        func = mod.lookup_function(func_name)
        _invoke(ctx.vm_context, device, func, kernel_inputs, kernel_outputs)

    if run_bench:
        inputs = [inp.numpy() for inp in kernel_inputs]
        benchmark_results = bench.benchmark_module(
            mod,
            entry_function=func_name,
            device=device,
            inputs=inputs,
            **benchmark_flags,
        )
        _print_bench_result(benchmark_results, bench_file)


def safe_subs(input: Any, subs: List[Tuple[IndexExpr, IndexExpr]]) -> Any:
    """
    Substitute input using provided `subs` list if input is sympy object.
    Otherwise return input unchanged.
    """
    if isinstance(input, (sympy.Basic, IndexSequence)):
        return input.subs(subs)

    return input


def subs_idxc(input: Any) -> Any:
    """
    Substitute input using IndexingContext if input is sympy object.
    Otherwise return input unchanged.
    """
    idxc = IndexingContext.current()
    return safe_subs(input, idxc.subs)
