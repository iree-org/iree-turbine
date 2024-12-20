# Copyright 2024 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
    Value,
)
from typing import Optional, Callable, Any, List, Tuple, Sequence
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexExpr, IndexingContext, IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from ..ops.wave_ops import (
    get_custom,
    Output,
    Write,
    MMA,
    CustomOp,
    Reduction,
    GetResult,
    ExtractSlice,
    IterArg,
    Reshape,
)
from ..lang.wave_types import IndexMapping
from .constraints import (
    Constraint,
    WorkgroupConstraint,
    HardwareConstraint,
    TilingConstraint,
    MMAType,
    MMAOperand,
)
from .assumptions import Assumption
import torch.fx as fx
import iree.turbine.kernel.lang as tkl
from pathlib import Path


import tempfile
from ...support.conversions import TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM
from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)

import sympy
import torch
from iree.compiler import compile_str
import iree.runtime as rt
import iree.runtime.benchmark as bench

# TODO: Monkey-patching f16 support, need to fix in iree.
import numpy
import ml_dtypes

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
                loops = structured_transform_ops.structured_match(
                    any_op_t(), target, ops=["scf.for"]
                )
                transform_d.apply_licm(loops)
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


def get_default_arch() -> str:
    """Return default ROCM architecture"""
    if not torch.cuda.is_available():
        return "cpu"
    device = torch.device("cuda")
    gcnArch = torch.cuda.get_device_properties(device).gcnArchName
    assert "gfx" in gcnArch, "Currently only support GFX/ROCm for get_default_arch."
    # The gcnArchName comes back like gfx90a:sramecc+:xnack.
    colon_pos = gcnArch.find(":")
    return gcnArch[0:colon_pos]


def get_default_run_config() -> dict[Any, Any]:
    """Return default config for running."""
    arch = get_default_arch()
    return {"backend": "rocm", "device": "hip", "target": arch}


def get_default_compile_config() -> dict[Any, Any]:
    """Return default config for compilation."""
    return {"backend": "rocm", "device": "hip", "target": "gfx942"}


def get_default_scheduling_params() -> dict[IndexSymbol, Any]:
    """Return default scheduling params."""
    # TODO: get values based on get_default_arch()
    return {
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }


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


def print_subgraph(trace: CapturedTrace, subgraph_name: str, custom_print: bool = True):
    """
    Prints a specific subgraphs of a trace.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for name, subgraph in trace.region_graph.subgraphs.items():
        if name == subgraph_name:
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


def remove_chained_getresult(trace: CapturedTrace):
    def is_chained_getresult(node: fx.Node) -> bool:
        custom = get_custom(node)
        return isinstance(custom, GetResult) and isinstance(
            get_custom(custom.value), GetResult
        )

    while removable_nodes := trace.walk(is_chained_getresult):
        for node in removable_nodes:
            get_custom(node).replace_all_uses_with(get_custom(node).value)
            get_custom(node).graph.erase_node(node)


def remove_chained_extractslice(trace: CapturedTrace):
    def is_chained_extractslice(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, ExtractSlice):
            return False
        register = get_custom(custom.register_)
        if not isinstance(register, ExtractSlice):
            return False
        return custom.rank == register.rank

    while removable_nodes := trace.walk(is_chained_extractslice):
        for node in removable_nodes:
            dst_extract = get_custom(node)
            src_extract = get_custom(dst_extract.register_)
            dst_extract.register_ = src_extract.register_
            new_offset = [
                dst_i + src_i
                for dst_i, src_i in zip(dst_extract.offset, src_extract.offset)
            ]
            dst_extract.update_arg("register_", src_extract.register_)
            dst_extract.update_arg("offset", new_offset)
            if len(src_extract.fx_node.users) == 0:
                get_custom(node).graph.erase_node(src_extract.fx_node)


def delinearize_index(
    index: IndexExpr, shape: list[int | IndexExpr]
) -> list[IndexExpr]:
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


def is_reshape_needed(
    node: CustomOp,
    node_vector_shapes: dict[IndexSymbol, int],
    vector_shapes: dict[IndexSymbol, int],
) -> bool:
    for dim in node.type.symbolic_shape:
        if dim not in vector_shapes:
            # Ignore nodes that are not used in both mmas.
            return False
        if node_vector_shapes[dim] != vector_shapes[dim]:
            return True
    return False


def get_mma_dimensional_mapping(
    trace: CapturedTrace,
    hardware_constraint: HardwareConstraint,
) -> tuple[
    dict[MMA, dict[IndexSymbol, int]], dict[MMA, dict[IndexSymbol, list[fx.Node]]]
]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2). We maintain this map per mma node and
    also update the vector_shapes of the mma node based on this information.
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMA)

    mapping: dict[MMA, dict[IndexSymbol, int]] = {}
    mma_nodes = trace.walk(is_mma)
    for node in mma_nodes:
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape
        k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        if custom not in mapping:
            mapping[custom] = {}
        mapping[custom][m] = MMAOperand.M
        mapping[custom][n] = MMAOperand.N
        mapping[custom][k] = MMAOperand.K
        custom.vector_shapes = {
            m: hardware_constraint.mma_matrix_shapes(custom.mma_type)[0],
            n: hardware_constraint.mma_matrix_shapes(custom.mma_type)[1],
            k: hardware_constraint.mma_matrix_shapes(custom.mma_type)[2],
        }
        if hardware_constraint.vector_shapes:
            custom.vector_shapes.update(hardware_constraint.vector_shapes)
        custom.reduction_dim = k

        # Since expansion proceeds bottom-up, we set the vector shapes
        # of the parent reduction to the vector shapes of the last MMA node.
        if hasattr(custom.graph, "parent_op"):
            reduction = get_custom(custom.graph.parent_op)
            reduction.vector_shapes = custom.vector_shapes

    # Determine if any reshapes are required. Reshapes are added for
    # chained matmuls when the vector shapes of the operands in one matmul
    # differ from those in another matmul. The mma_slices contain all the ops
    # in the backward slice of the lhs and rhs upto a previous mma (if one exists).
    # So we check for the previous node of the first operator in the slice to see
    # if it is an MMA and if so check if a reshape is required.
    def add_reshape_if_needed(mma: MMA, prev_mma: MMA, arg_index: int):
        with mma.graph.inserting_before(mma.fx_node):
            arg = mma.lhs if arg_index == 0 else mma.rhs
            arg = get_custom(arg)
            if is_reshape_needed(arg, mma.vector_shapes, prev_mma.vector_shapes):
                reshape = Reshape(arg.fx_node, prev_mma.vector_shapes).add_to_graph(
                    custom.graph
                )
                custom_reshape = get_custom(reshape)
                custom_reshape.vector_shapes = custom.vector_shapes
                custom.update_arg(arg_index, reshape)

    def find_mma_in_slice(node: CustomOp) -> Optional[MMA]:
        """
        Find the closest mma by iterating through the backward slice of a node
        in reverse.
        """
        slice = list(capture_backward_slice(node))
        for arg in reversed(slice):
            prev_mma = get_custom(arg)
            if isinstance(prev_mma, MMA):
                return prev_mma
        return None

    # Look in the backward slices of both the LHS and RHS to find
    # mmas. If found, add reshapes if necessary.
    for mma in mma_nodes:
        custom_mma = get_custom(mma)
        prev_mma = find_mma_in_slice(custom_mma.lhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 0)
        prev_mma = find_mma_in_slice(custom_mma.rhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 1)

    return mapping


def get_hardware_vector_size(
    dim: IndexSymbol,
    hardware_constraint: HardwareConstraint,
    mma_indices: dict[IndexSymbol, int],
) -> int:
    """
    Given a hardware constraint, return the vector sizes for the given dimension.
    This could be a hardware specific vector size or a user specified vector size.
    """
    if mma_indices:
        vector_size = hardware_constraint.mma_matrix_shapes[mma_indices[dim]]
    else:
        vector_size = hardware_constraint.vector_shapes[dim]
    return vector_size


def get_hardware_vector_map(constraints: list[Constraint]) -> dict[IndexSymbol, int]:
    """
    Given a list of constraints, looks for hardware constraint and return a map
    containing dim's and their respective vector sizes.
    """
    vector_map = {}
    for c in constraints:
        if isinstance(c, HardwareConstraint):
            vector_map = c.vector_shapes
            break
    return vector_map


def remove_global_indexing(
    index: dict[IndexSymbol, IndexSequence], constraints: list[Constraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    workgroup and induction level indexing. This is necessary for writes to shared memory
    that operate on promoted memory.
    """
    tiling_constraints = [c for c in constraints if isinstance(c, TilingConstraint)]
    workgroup_ids = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
    subs = {w: 0 for w in workgroup_ids}

    new_index = {key: safe_subs(index[key], subs) for key in index}
    for key in new_index:
        for constraint in tiling_constraints:
            new_index[key] = new_index[key].subs({constraint.induction_var: 0})
    return new_index


def is_shared_mem_access(custom: "CustomOp") -> bool:
    return custom.memory_type.address_space == SHARED_ADDRESS_SPACE


def align_index_vars(
    index: dict[IndexSymbol, IndexSequence], constraints: list[Constraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function aligns index vars with Workgroup/Tiling constraints so it never
    need partial reads/writes.
    """
    key_subs = {
        c.dim: (c.count * c.tile_size)
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
        and subs_idxc(c.dim) != subs_idxc(c.count * c.tile_size)
    }
    return {safe_subs(key, key_subs): index[key] for key in index}


def _invoke(vm_context, device, entry_function, inputs, outputs, dynamic_dims):
    arg_list = rt.VmVariantList(len(inputs) + len(dynamic_dims))
    ret_list = rt.VmVariantList(len(outputs))

    for input in inputs:
        if isinstance(input, torch.Tensor):
            input_cpu = input.cpu().contiguous()
            device_array = rt.asdevicearray(device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    for dynamic_dim in dynamic_dims:
        if isinstance(dynamic_dim, int):
            arg_list.push_int(dynamic_dim)
        else:
            raise ValueError(f"Unsupported dynamic dim type: {type(dynamic_dim)}")

    vm_context.invoke(entry_function, arg_list, ret_list)

    for i, ret in enumerate(outputs):
        device_buffer_view = rt.HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
        device_array = rt.DeviceArray(device, device_buffer_view)

        # TODO: Make to_host accept out array/buffer, so we can avoid extra data copy.
        host_array = device_array.to_host()

        # Convert to torch tensor without actually importing torch.
        ret[:] = type(ret)(host_array)


def _inplace_invoke(vm_context, device, entry_function, inputs, outputs, dynamic_dims):
    linearized_arg_len = len(inputs) + len(outputs) + len(dynamic_dims)
    # ret_list is 0 because we modify/write result in place.
    arg_list = rt.VmVariantList(linearized_arg_len)
    ret_list = rt.VmVariantList(0)

    def push_tensor_to_arg_list(arg_tensor: torch.Tensor):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        capsule = arg_tensor.__dlpack__(None)
        arg_tensor_bv = device.from_dlpack_capsule(capsule)
        arg_list.push_ref(arg_tensor_bv)

    # Linearize arguments, In linearized arg_list, we first push in all inputs,
    # then all the outputs, and lastly all the dynamic dims.
    for input in inputs:
        if isinstance(input, torch.Tensor):
            push_tensor_to_arg_list(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    for output in outputs:
        if isinstance(input, torch.Tensor):
            push_tensor_to_arg_list(output)
        else:
            raise ValueError(f"Unsupported output type: {type(output)}")
    for dynamic_dim in dynamic_dims:
        if isinstance(dynamic_dim, int):
            arg_list.push_int(dynamic_dim)
        else:
            raise ValueError(f"Unsupported dynamic dim type: {type(dynamic_dim)}")

    vm_context.invoke(entry_function, arg_list, ret_list)


def _read_file(name, mode):
    with open(name, mode) as file:
        data = file.read()
    return data


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


def get_device_uuid(input_tensors: list[torch.Tensor]) -> tuple[int, str]:
    """
    Checks all torch.Tensor are on the same device, and get UUID from Torch device.
    """
    device_list = [
        input.device for input in input_tensors if isinstance(input, torch.Tensor)
    ]
    if len(set(device_list)) != 1:
        raise ValueError(f"Found multiple device on input tensors:{set(device_list)}")
    device = device_list[0]
    if device.type != "cuda":
        raise ValueError("Expected all argument tensors to be in GPU.")
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return uuid


def compile_to_vmfb(
    asm: str,
    config: dict[str, str],
    run_bench: bool = False,
):

    backend = config["backend"]
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

    preprocessing_pipeline = config.get("iree_preprocessing_pass_pipeline", None)
    if preprocessing_pipeline is not None:
        flags.append(f"--iree-preprocessing-pass-pipeline={preprocessing_pipeline}")

    if "dump_intermediates" in config:
        intermediates_path = config.get("dump_intermediates")
        flags.append(
            f"--iree-hal-dump-executable-intermediates-to={intermediates_path}"
        )

    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        if bench_batch_size is not None:
            flags.append(
                f"--iree-hal-benchmark-dispatch-repeat-count={bench_batch_size}"
            )

    res = compile_str(asm, target_backends=[backend], extra_args=flags)
    return res


def invoke_vmfb(
    vmfb: bytes,
    func_name: str,
    config: dict[str, str],
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    kernel_dynamic_dims: list[int] = [],
    run: bool = False,
    run_bench: bool = False,
    inplace: bool = False,
):

    device = config["device"]
    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        bench_repetitions = config.get("benchmark_repetitions", None)
        bench_file = config.get("benchmark_results_file", None)

        benchmark_flags = {}

        if bench_batch_size is not None:
            benchmark_flags["batch_size"] = int(bench_batch_size)

        if bench_repetitions is not None:
            benchmark_flags["benchmark_repetitions"] = int(bench_repetitions)

    if not (run or run_bench):
        return

    if inplace:
        # Select device as the GPU, where input tensors are coming from.
        device_uuid = get_device_uuid(kernel_inputs + kernel_outputs)
        device = f"{device}://GPU-{device_uuid}"
    rt_config = rt.Config(device)
    device = rt_config.device
    vm_instance = rt_config.vm_instance
    mod = rt.VmModule.copy_buffer(vm_instance, vmfb)

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
        if inplace:
            _inplace_invoke(
                ctx.vm_context,
                device,
                func,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
            )
        else:
            _invoke(
                ctx.vm_context,
                device,
                func,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
            )

    if run_bench:
        bench_with_constant_weights = config.get("bench_with_constant_weights", False)
        tempfiles = []
        inputs = []
        all_inputs = kernel_inputs + kernel_outputs if inplace else kernel_inputs
        if bench_with_constant_weights:
            for inp in all_inputs:
                if isinstance(inp, torch.Tensor):
                    inputs.append(
                        "x".join(
                            [str(x) for x in inp.shape]
                            + [TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM[inp.dtype]]
                        )
                    )
                elif isinstance(inp, int):
                    inputs.append(f"1xi32={inp}")
                else:
                    raise NotImplementedError("Unsupported input type.")
        else:
            for inp in all_inputs:
                if isinstance(inp, torch.Tensor):
                    tf = tempfile.NamedTemporaryFile(suffix=".npy")
                    inp = inp.cpu()
                    if inp.dtype == torch.bfloat16:
                        inp = (
                            inp.view(dtype=torch.uint16)
                            .numpy()
                            .view(dtype=ml_dtypes.bfloat16)
                        )
                    else:
                        inp = inp.numpy()
                    numpy.save(tf, inp)
                    tempfiles.append(tf)
                    inputs.append("@" + tf.name)
                elif isinstance(inp, int):
                    inputs.append(f"1xi32={inp}")
                else:
                    raise NotImplementedError("Unsupported input type.")

        benchmark_results = bench.benchmark_module(
            mod,
            entry_function=func_name,
            device=device,
            inputs=inputs,
            **benchmark_flags,
        )
        _print_bench_result(benchmark_results, bench_file)


def compile_and_invoke(
    asm: str,
    func_name: str,
    config: dict[str, str],
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    kernel_dynamic_dims: list[int] = [],
    run: bool = False,
    run_bench: bool = False,
    inplace: bool = False,
):
    compiled_wave_vmfb = compile_to_vmfb(asm, config, run_bench)
    invoke_vmfb(
        compiled_wave_vmfb,
        func_name,
        config,
        kernel_inputs,
        kernel_outputs,
        kernel_dynamic_dims,
        run,
        run_bench,
        inplace,
    )


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


def graph_copy(graph: fx.Graph) -> tuple[fx.Graph, dict[fx.Node, fx.Node]]:
    """
    Copy the graph and return the new graph with the nodes in node_map.
    Also return the mapping of old nodes to new nodes.
    """
    new_graph = fx.Graph()
    node_map = {}
    for node in graph.nodes:
        custom = get_custom(node)
        new_node = custom.copy(
            new_graph=new_graph,
            arg_transform=lambda x: node_map[x] if x in node_map else x,
        )
        node_map[node] = new_node.fx_node
    return new_graph, node_map


def erase_graph(graph: fx.Graph):
    """
    Erase all nodes in the graph.
    """
    for node in reversed(graph.nodes):
        for user in node.users:
            graph.erase_node(user)
        graph.erase_node(node)


def get_users(
    node: fx.Node, reduction: fx.Node = None
) -> tuple[list[fx.Node], fx.Node]:
    """
    Return the users of a node, propagating through reductions.
    """
    users = []
    for user in node.users:
        custom = user
        if not isinstance(custom, CustomOp):
            custom = get_custom(user)
        if isinstance(custom, Reduction):
            # Map init arg to iter arg
            reduction = custom
            init_arg_idx = custom.init_args.index(node)
            graph = custom.get_root_graph().subgraphs[custom.subgraph_name]
            users.append(custom.iter_args(graph)[init_arg_idx])
            continue
        if isinstance(custom, Output):
            # Map output to get result
            return_vals = custom.return_vals[0]
            parent_reduction = custom.graph.parent_op
            if not isinstance(return_vals, (list, tuple)):
                users.append(next(iter(parent_reduction.users)))
            else:
                # Handles case where DCE eliminate unused GetResult.
                get_results = {
                    get_custom(x).res_idx: x
                    for x in parent_reduction.users
                    if isinstance(get_custom(x), GetResult)
                }
                output_idx = return_vals.index(node)
                # Sometime IterArg only used within the tkw.Reduction region
                if output_idx in get_results:
                    users.append(get_results[output_idx])
            continue
        users.append(user)
    return users, reduction


def get_inputs(
    node: fx.Node, reduction: fx.Node = None
) -> tuple[list[fx.Node], fx.Node]:
    """
    Return the inputs of a node, propagating through reductions.
    """
    inputs = []
    custom = get_custom(node)
    if isinstance(custom, IterArg):
        # Map iter args to init args
        local_reduction = reduction
        if reduction is None:
            local_reduction = custom.parent_op()
        iter_arg_idx = custom.get_iter_idx()
        inputs.append(local_reduction.init_args[iter_arg_idx])
    elif isinstance(custom, GetResult):
        reduction = get_custom(custom.value)
        assert isinstance(reduction, Reduction), "GetResult must be used by a Reduction"
        # Map get result to output
        reduction_subgraph = reduction.graph.subgraphs[reduction.subgraph_name]
        inputs.append(reduction.outputs(reduction_subgraph)[custom.res_idx])
    elif isinstance(custom, Reduction):
        reduction_subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        inputs.append(custom.outputs(reduction_subgraph))
    else:
        # Default handling for other ops.
        for input in node.all_input_nodes:
            inputs.append(input)
    return inputs, reduction


def bfs(
    node: fx.Node,
    get_neighbors: Callable[[fx.Node, fx.Node], list[fx.Node]],
    filter_fn: Callable[[fx.node], bool],
) -> set[fx.Node]:
    """
    Run BFS on the graph. The filter function is not applied to
    the incoming node.
    """
    visited: set[fx.Node] = set()
    queue: list[fx.Node] = []
    visited.add(node)
    queue.append(node)
    reduction = None
    while queue:
        s = queue.pop(0)
        neighbors, reduction = get_neighbors(s, reduction)
        for neighbor in neighbors:
            if neighbor not in visited and filter_fn(neighbor):
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def capture_forward_slice(
    node: fx.Node,
    filter_fn: Callable[[fx.node], bool] = lambda x: True,
) -> set[fx.Node]:
    """
    Run BFS on the graph to capture the forward slice of a node.
    """
    return bfs(node, lambda x, y: get_users(x, y), filter_fn)


def capture_backward_slice(
    node: fx.Node, filter_fn: Callable[[fx.node], bool] = lambda x: True
) -> set[fx.Node]:
    """
    Capture backward slice from a node and return the tree.
    Assumes graph is directed.
    """
    return bfs(node, lambda x, y: get_inputs(x, y), filter_fn)


def capture_mma_slices(mma: MMA) -> dict[IndexSymbol, list[fx.Node]]:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    """
    mma_slices = {x: [] for x in [MMA_LHS, MMA_RHS, MMA_ACC]}
    is_not_mma = lambda x: not isinstance(get_custom(x), MMA)
    mma_slices[MMA_LHS] += capture_backward_slice(mma.lhs, is_not_mma)
    mma_slices[MMA_RHS] += capture_backward_slice(mma.rhs, is_not_mma)
    mma_slices[MMA_ACC] += capture_forward_slice(mma.fx_node, is_not_mma).union(
        capture_backward_slice(mma.acc, is_not_mma)
    )
    return mma_slices


def specialize_index_sequence(
    index_seq: IndexSequence,
    mma_slices: dict[IndexSymbol, list[fx.Node]],
    custom: CustomOp,
) -> IndexSequence:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    If the node is not used as any of the operands, return the original index sequence
    with all the MMA symbols zeroed out.
    """
    if isinstance(custom, MMA):
        return index_seq
    operand_map = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 0}
    for key in mma_slices:
        if custom.fx_node in mma_slices[key]:
            operand_map[key] = 1
            return index_seq.subs(operand_map)
    return index_seq.subs(operand_map)


def find_index_bounds(
    constraints: list[Constraint], index: dict[IndexExpr, IndexExpr]
) -> Optional[list[IndexExpr]]:
    bounds = []
    for constraint in constraints:
        if not isinstance(constraint, (WorkgroupConstraint, TilingConstraint)):
            continue

        dim = constraint.dim
        if dim not in index:
            continue

        work_size = constraint.count * constraint.tile_size
        if subs_idxc(work_size) == subs_idxc(dim):
            continue

        bounds.append(dim)

    if len(bounds) == 0:
        return None

    return bounds


def get_induction_variable(
    reduction: Reduction, constraints: list[Constraint]
) -> IndexSymbol:
    induction_var = None
    for constraint in constraints:
        if (
            isinstance(constraint, TilingConstraint)
            and reduction.axis == constraint.dim
        ):
            induction_var = constraint.induction_var
            break
    else:
        raise ValueError(f"Could not find induction variable for reduction {reduction}")
    return induction_var


def get_tiling_constraint(
    reduction: Reduction, constraints: list[Constraint]
) -> TilingConstraint:
    for constraint in constraints:
        if (
            isinstance(constraint, TilingConstraint)
            and reduction.axis == constraint.dim
        ):
            return constraint
    else:
        raise ValueError(f"Could not find tiling constraint for reduction {reduction}")


def get_hardware_constraint(constraints: list[Constraint]) -> HardwareConstraint:
    for constraint in constraints:
        if isinstance(constraint, HardwareConstraint):
            return constraint
    else:
        raise ValueError(f"Could not find hardware constraint in {constraints}")


def get_workgroup_constraints(
    constraints: list[Constraint],
) -> list[WorkgroupConstraint]:
    return [x for x in constraints if isinstance(x, WorkgroupConstraint)]


def replace_uses_in(users: dict[fx.Node, list[CustomOp]], old: CustomOp, new: fx.Node):
    """
    Replace all uses of `old` with `new` in the list of users.
    """
    for user in users[old]:
        for i, arg in enumerate(user.fx_node.args):
            if arg == old.fx_node:
                user.update_arg(i, new)


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)


def get_mfma_load_elems_per_thread(mfma_variant: MMAType) -> int:
    match mfma_variant:
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 4
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 8
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 8


def get_mfma_store_elems_per_thread(mfma_variant: MMAType) -> int:
    match mfma_variant:
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 16
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 4
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 16


def all_equal(input_list: list[Any]) -> bool:
    if len(input_list) == 0:
        return True
    return all(elem == input_list[0] for elem in input_list)


DEFAULT_GPU_DEVICE = None


def get_default_gpu_device_name() -> str:
    if DEFAULT_GPU_DEVICE is None:
        return "cuda"

    return f"cuda:{DEFAULT_GPU_DEVICE}"


def get_default_device() -> str:
    return get_default_gpu_device_name() if torch.cuda.is_available() else "cpu"


def to_default_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(get_default_device())


def device_randn(*args, **kwargs):
    return to_default_device(torch.randn(*args, **kwargs))


def device_randint(*args, **kwargs):
    return to_default_device(torch.randint(*args, **kwargs))


def device_randperm(*args, **kwargs):
    return to_default_device(torch.randperm(*args, **kwargs))


def device_zeros(*args, **kwargs):
    return to_default_device(torch.zeros(*args, **kwargs))


def get_assumptions(constraints: list[Constraint]) -> list[Assumption]:
    assumptions: list[Assumption] = []
    for constraint in constraints:
        if isinstance(constraint, Assumption):
            assumptions.append(constraint)
    return assumptions


def evaluate_with_assumptions(constraints: list[Constraint], expr: IndexExpr) -> bool:
    """
    Evalutes whether the expression is true given the assumptions.
    To do this, we solve the assumptions and target expression and check
    that the result is in the assumptions.
    """
    facts = [subs_idxc(x.expr) for x in get_assumptions(constraints)]
    result = sympy.solve(facts + [expr])
    # Solve returns a false if the inequalities are not consistent.
    if isinstance(result, sympy.logic.boolalg.BooleanAtom):
        return False
    return True if any([result.equals(x) for x in facts]) else None


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def _simplify_sympy_expr(expr: IndexExpr) -> IndexExpr:
    """Apply custom sympy simplifications"""

    def check_mul(mul):
        ret = None
        for arg in mul.args:
            if arg.is_number:
                if arg < 0:
                    return None

                if ret is not None:
                    return None

                ret = arg
                continue

            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None

            if not arg.is_nonnegative:
                return None

        return ret

    def transform_mod(expr):
        """Move constant outside of Mod expr

        Example:
        (floor(a) * 4 + 3) % 16 -> (floor(a) * 4) % 16 + 3
        """
        if not isinstance(expr, sympy.Mod):
            return None

        p, q = expr.args
        if not q.is_number or q < 0:
            return None

        if not isinstance(p, sympy.Add):
            return None

        c = None
        terms = []
        mult = None
        for arg in p.args:
            if arg.is_number:
                if c is not None:
                    return None

                c = arg
                continue

            if not isinstance(arg, sympy.Mul):
                return None

            m = check_mul(arg)
            if (m is None) or (q % m != 0):
                return None

            mult = m if (mult is None) or (m < mult) else mult
            terms.append(arg)

        if c >= mult:
            return None

        return (sum(terms) % q) + c

    def check_mul_rational(mul):
        ret = None
        for arg in mul.args:
            if isinstance(arg, sympy.Rational):
                if ret is not None:
                    return None

                if arg.p < 0 or arg.q < 0:
                    return None

                ret = arg
                continue

            if not (isinstance(arg, (sympy.floor, sympy.Mod)) or arg.is_integer):
                return None

            if not arg.is_nonnegative:
                return None

        return ret

    def transform_floor(expr):
        """Simplify rational addition inside floor expr

        Example:
        floor(floor(a)/3 + 1/6) -> floor(floor(a)/3)
        """
        if not isinstance(expr, sympy.floor):
            return None

        expr = expr.args[0]
        if not isinstance(expr, sympy.Add):
            return None

        c = None
        for arg in expr.args:
            if isinstance(arg, sympy.Rational):
                if c is not None:
                    return None

                c = arg

        if c is None:
            return None

        terms = []
        for arg in expr.args:
            if isinstance(arg, sympy.Rational):
                continue

            if not isinstance(arg, sympy.Mul):
                return None

            r = check_mul_rational(arg)
            if r is None or r.p != 1:
                return None

            if r <= c:
                return None

            terms.append(arg)

        return sympy.floor(sum(terms))

    expr = expr.replace(lambda e: transform_mod(e) is not None, transform_mod)
    expr = expr.replace(lambda e: transform_floor(e) is not None, transform_floor)
    return sympy.simplify(expr)


def approximate_difference(
    expr: IndexExpr, vars: list[IndexSymbol], elements_per_thread: int
) -> bool:
    """
    During the contiguity check, we take a unit step in the fastest changing
    dimension (j -> j + 1) and we compute f(j + 1) - f(j) to see if it is 1.
    In general, we will end up with expressions of the form
    g(x + eps) - g(x) where x = h(j) and eps is a rational of the form 1/q.
    We can use q to determine if the mapping is contiguous as follows

    if q is divisible by elements_per_thread (dimensions where we have not applied the unit step), or
    if eps is 1 (corresponds to the dimension where we have applied the unit step)
    then the mapping is contiguous.

    The mapping function f(j) will be non-linear in general, and so the difference
    of 1 will be transformed to different constant values based on the function.
    But, if we recover a value of 1, we can assume that the function preserves
    the difference.

    In this function we do a pre-order traversal of the expression to obtain
    the value of the constant eps.
    """
    if expr.is_number:
        return expr
    new_vars, new_exprs = sympy.cse(expr)
    new_expr = new_exprs[0] if new_vars else expr
    new_vars = [x[0] for x in new_vars] if new_vars else vars
    for arg in sympy.preorder_traversal(new_expr):
        if isinstance(arg, sympy.Add):
            if all([x in arg.args for x in new_vars]):
                constant = [x for x in arg.args if x not in new_vars][0]
                if not isinstance(constant, sympy.Rational):
                    return expr
                if constant.p != 1:
                    return expr
                if constant.q == 1:
                    return 1
                return 0 if constant.q % elements_per_thread == 0 else expr
    return expr


def check_is_mapping_contiguous(
    mapping: IndexMapping,
    symbolc_shape: tuple[IndexExpr, ...],
    index: tuple[IndexExpr, ...],
    elements_per_thread: int | IndexExpr,
    is_read: bool,
) -> bool:
    """Check if mapping can be lowered to contiguous vector ops instead of gathers/scatters"""
    elements_per_thread = subs_idxc(elements_per_thread)
    if elements_per_thread == 1:
        return True

    # TODO: Better dyn vals analysis.
    if mapping.num_dynamic_vals != 0:
        return False

    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        index_mapping = mapping.map_input_indices(symbolc_shape)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolc_shape)

    index_mapping = tuple(subs_idxc(i) for i in index_mapping)
    iters = mapping.iters

    subs = [(sym, sym + int(i == len(iters) - 1)) for i, sym in enumerate(iters)]
    diff = [
        approximate_difference(
            index_mapping[i].subs(subs) - index_mapping[i],
            list(iters.keys())[-1:],
            elements_per_thread,
        )
        for i in range(len(index_mapping))
    ]

    expected_diff = [0] * len(index_mapping)
    expected_diff[-1] = 1

    return diff == expected_diff


def get_largest_index_and_size(indices: dict[IndexExpr, IndexSequence]):
    """
    This function takes in indices of a Node, extract their sizes
    into a list, and then returns the dimension with the largest size.
    In case of ties, it picks the fastest changing dimension.
    """

    sorted_values = sorted(
        [
            (i, dim, subs_idxc(index.size))
            for i, (dim, index) in enumerate(indices.items())
        ],
        # x[0] is the index of the dimension.
        # x[2] is the size of the dimension.
        # We want to sort in descending order, first by size, then by index.
        # (pick the largest size with the largest index).
        key=lambda x: (-x[2], -x[0]),
    )
    return sorted_values[0][1:]
