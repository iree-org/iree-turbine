# Copyright 2024 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

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
    Conditional,
    CustomOp,
    ExtractSlice,
    GetResult,
    IterArg,
    MMA,
    NestedRegionOp,
    Output,
    Placeholder,
    Read,
    Reduction,
    Reshape,
    SetSymbol,
    SharedMemoryBarrier,
    Write,
    get_custom,
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


from ...support.conversions import TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM
from .profiling import benchmark_module
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

# TODO: Monkey-patching f16 support, need to fix in iree.
import numpy
import ctypes
from dataclasses import dataclass
import glob
import os
import pickle


@dataclass
class KernelLaunchInfo:
    grid: tuple[int] = None
    blocks: tuple[int] = None
    shared_memory_bytes: int = 0
    func_name: str = ""


def try_apply_pass(
    p,
    trace: CapturedTrace,
    print_ir_before: Sequence[str] = [],
    print_ir_after: Sequence[str] = [],
):
    if "all" in print_ir_before or p.__name__ in print_ir_before:
        print(f"***Before {p.__name__}***\n")
        print_trace(trace)
    try:
        p()
    except Exception:
        print(f"Error in pass: {p.__name__}\n")
        print_trace(trace)
        raise
    if "all" in print_ir_after or p.__name__ in print_ir_after:
        print(f"***After {p.__name__}***\n")
        print_trace(trace)


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
    # Print func name before running
    print(f"{func.__name__}")
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
    for name, subgraph in reversed(list(trace.region_graph.subgraphs.items())):
        if name == trace.root_graph:
            name = f"{name} [root]"
        print(f"{name}:\n")
        print_graph(subgraph)
        if custom_print:
            print("Custom format:")
            for node in subgraph.nodes:
                print(get_custom(node))


def print_subgraph(trace: CapturedTrace, subgraph_name: str, custom_print: bool = True):
    """
    Prints a specific subgraphs of a trace.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
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

    def is_global_write(node: fx.Node) -> bool:
        custom = get_custom(node)
        return isinstance(custom, Write) and (
            subs_idxc(custom.type.address_space)
            in [GLOBAL_ADDRESS_SPACE, tkl.AddressSpace.GLOBAL_MEMORY.value]
        )

    def has_nested_writes(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, NestedRegionOp):
            return False

        subgraph = custom.graph.subgraphs[custom.subgraph_name]
        for node in subgraph.nodes:
            if is_global_write(node) or has_nested_writes(node):
                return True

        return False

    def is_removable_operator(node: fx.Node) -> bool:
        custom = get_custom(node)

        if (
            custom.users
            or isinstance(custom, (Output, SetSymbol, SharedMemoryBarrier))
            or is_global_write(node)
        ):
            return False

        if has_nested_writes(node):
            return False

        return True

    while removable_nodes := trace.walk(is_removable_operator):
        for node in removable_nodes:
            get_custom(node).erase()


def move_node_after(src_node: fx.Node, anchor: fx.Node):
    """
    Moves src_node into a location after a given anchor node.
    This function will invalidate "src_node" and return the
    newly copied/"moved" node.
    """
    custom_src = get_custom(src_node)
    moved_src = custom_src.copy(anchor=(anchor)).fx_node
    custom_src.replace_all_uses_with(moved_src)
    src_name = src_node.name
    src_node.graph.erase_node(src_node)
    moved_src.name = src_name
    return moved_src


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
                    mma.graph
                )
                custom_reshape = get_custom(reshape)
                custom_reshape.vector_shapes = mma.vector_shapes
                mma.update_arg(arg_index, reshape)

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
            new_dim = new_index[key]
            if sympy.sympify(new_dim.start).has(constraint.induction_var):
                new_dim = new_dim.subs({constraint.induction_var: 0})
                new_dim.start = new_dim.start - constraint.start
                new_index[key] = new_dim
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


_dl_tensor_name = ctypes.create_string_buffer(b"dltensor")
_set_capsule_name = ctypes.pythonapi.PyCapsule_SetName


def _inplace_invoke(vm_context, device, entry_function, inputs, outputs, dynamic_dims):
    linearized_arg_len = len(inputs) + len(outputs) + len(dynamic_dims)
    # ret_list is 0 because we modify/write result in place.
    arg_list = rt.VmVariantList(linearized_arg_len)
    ret_list = rt.VmVariantList(0)

    def push_tensor_to_arg_list(arg_tensor: torch.Tensor):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        capsule = torch.to_dlpack(arg_tensor)
        arg_tensor_bv = device.from_dlpack_capsule(capsule)

        # IREE runtime renames capsule to "dltensor_used" for some reason, but
        # only deletes capsules with "dltensor" name, which is causing a memory
        # leak.
        _set_capsule_name(ctypes.py_object(capsule), _dl_tensor_name)
        arg_list.push_ref(arg_tensor_bv)

    # Linearize arguments, In linearized arg_list, we first push in all inputs,
    # then all the outputs, and lastly all the dynamic dims.
    for input in inputs:
        if isinstance(input, torch.Tensor):
            push_tensor_to_arg_list(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    for output in outputs:
        if isinstance(output, torch.Tensor):
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


@functools.lru_cache
def get_device_uuid(device_list: list[str], device_str: str) -> tuple[int, str]:
    """
    Checks all torch.Tensor are on the same device, and get UUID from Torch device.
    """
    if len(set(device_list)) != 1:
        raise ValueError(f"Found multiple device on input tensors:{set(device_list)}")
    device = device_list[0]
    if device.type != "cuda":
        raise ValueError("Expected all argument tensors to be in GPU.")
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    device_str = f"{device_str}://GPU-{uuid}"
    return device_str


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

    if config.get("gpu-native-math-precision", False):
        # Polynomial approximation passes in MLIR/IREE often generate
        # suboptimal code with redundant clamps and fptosi. This flag
        # allows us to skip unnecessary approx for GPU.
        flags.append("--iree-codegen-gpu-native-math-precision=true")

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

    if binaries_path := config.get("dump_binaries", None):
        flags.append(f"--iree-hal-dump-executable-binaries-to={binaries_path}")

    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        if bench_batch_size is not None:
            flags.append(
                f"--iree-hal-benchmark-dispatch-repeat-count={bench_batch_size}"
            )

    res = compile_str(asm, target_backends=[backend], extra_args=flags)
    return res


# Cache for the system context and vm function.
RUNTIME_CACHE: dict[str, tuple[rt.SystemContext, rt.VmFunction]] = {}


def invoke_with_wave_runtime(
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    kernel_dynamic_dims: list[int],
    kernel_hash: str,
    kernel_info: KernelLaunchInfo,
):
    """
    Invokes the kernel with the wave runtime.
    """
    import wave_runtime
    from .cache import WAVE_RUNTIME_DIR, CACHE_BASE_DIR

    # Get the path to the binary.
    if kernel_hash:
        binary = str(CACHE_BASE_DIR / kernel_hash / kernel_hash) + ".hsaco"
    else:
        binary = glob.glob(str(WAVE_RUNTIME_DIR / "*.hsaco"))[0]

    # Populate all the information required to launch the kernel.
    hash_str = "" if not kernel_hash else kernel_hash
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        binary,
        kernel_info.func_name,
        hash_str,
        kernel_info.shared_memory_bytes,
        kernel_info.grid[0],
        kernel_info.grid[1],
        kernel_info.grid[2],
        kernel_info.blocks[0],
        kernel_info.blocks[1],
        kernel_info.blocks[2],
    )

    # Ensure that the tensors are contiguous.
    kern_args = []
    for arg_tensor in kernel_inputs + kernel_outputs:
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        kern_args.append(arg_tensor.data_ptr())

    kernel_args = wave_runtime.Int64Vector(kern_args)
    dyn_dims = wave_runtime.Int64Vector(list(kernel_dynamic_dims))
    # Launch the kernel.
    wave_runtime.launch(kernel_launch_info, kernel_args, dyn_dims)


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
    kernel_hash: Optional[str] = None,
    kernel_launch_info: Optional[KernelLaunchInfo] = None,
):
    wave_runtime_launcher = config.get("wave_runtime", None)
    if wave_runtime_launcher:
        invoke_with_wave_runtime(
            kernel_inputs,
            kernel_outputs,
            kernel_dynamic_dims,
            kernel_hash,
            kernel_launch_info,
        )
        return

    device = config["device"]
    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        bench_repetitions = config.get("benchmark_repetitions", None)
        bench_file = config.get("benchmark_results_file", None)

        benchmark_flags = {}

        # If we use 1000 for bench_batch_size during compilation, and set this batch size to 1,
        # then the latency is in milliseconds.
        benchmark_flags["batch_size"] = 1

        if bench_repetitions is not None:
            benchmark_flags["benchmark_repetitions"] = int(bench_repetitions)

    if not (run or run_bench):
        return

    if inplace:
        # Select device as the GPU, where input tensors are coming from.
        device_list = tuple(
            input.device
            for input in kernel_inputs + kernel_outputs
            if isinstance(input, torch.Tensor)
        )
        device = get_device_uuid(device_list, device)
    rt_config = rt.Config(device)
    device = rt_config.device
    vm_instance = rt_config.vm_instance

    if kernel_hash and kernel_hash in RUNTIME_CACHE:
        ctx, func = RUNTIME_CACHE[kernel_hash]
    else:
        mod = rt.VmModule.copy_buffer(vm_instance, vmfb)
        vm_modules = [
            mod,
            rt.create_hal_module(vm_instance, device),
        ]
        ctx = rt.SystemContext(
            vm_modules=vm_modules,
            config=rt_config,
        )
        func = mod.lookup_function(func_name)
        if kernel_hash:
            RUNTIME_CACHE[kernel_hash] = (ctx, func)

    if run:
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
        benchmark_results = benchmark_module(
            kernel_inputs,
            kernel_outputs,
            kernel_dynamic_dims,
            config,
            inplace,
            mod,
            entry_function=func_name,
            device=device,
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
            graph = custom.get_root_graph().subgraphs[custom.subgraph_name]
            if node in custom.init_args:
                init_arg_idx = custom.init_args.index(node)
                users.append(custom.iter_args(graph)[init_arg_idx])
            else:
                assert node in custom.implicit_captures
                for outside_node in graph.nodes:
                    if outside_node.meta.get("lifted", None) == node:
                        users.append(outside_node)
                        break
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
        if isinstance(custom, Conditional):
            if node == custom.condition:
                users.append(user)
            else:
                subgraph = custom.graph.subgraphs[custom.subgraph_name]
                var = custom.get_captured_fx_node(subgraph, node)
                assert var is not None, "Invalid captured var"
                for u in var.users:
                    users.append(u)

            continue

        users.append(user)
    return users, reduction


def propagate_placeholders(n):
    """
    Returns the captured node of a placeholder if it exists.
    """
    c = get_custom(n)
    if isinstance(c, Placeholder):
        p = c.get_captured_fx_node()
        if p is not None:
            return p
    return n


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
        iter_arg_idx = custom.iter_idx
        inputs.append(local_reduction.init_args[iter_arg_idx])
    elif isinstance(custom, GetResult):
        reduction = get_custom(custom.value)
        assert isinstance(reduction, Reduction), "GetResult must be used by a Reduction"
        # Map get result to output
        reduction_subgraph = reduction.graph.subgraphs[reduction.subgraph_name]
        if len(reduction.init_args) == 1:
            outputs = reduction.outputs(reduction_subgraph)
            if isinstance(outputs, Sequence):
                inputs += outputs
            else:
                inputs.append(outputs)
        else:
            inputs.append(reduction.outputs(reduction_subgraph)[custom.res_idx])
    elif isinstance(custom, Reduction):
        reduction_subgraph = custom.get_root_graph().subgraphs[custom.subgraph_name]
        inputs.append(custom.outputs(reduction_subgraph))
    else:
        # Default handling for other ops.
        for input in node.all_input_nodes:
            inputs.append(input)

    inputs = [propagate_placeholders(i) for i in inputs]
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
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 8
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K8_F16
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
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 4
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K8_F16
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


def device_arange(*args, **kwargs):
    return to_default_device(torch.arange(*args, **kwargs))


def device_empty(*args, **kwargs):
    return to_default_device(torch.empty(*args, **kwargs))


def device_full(*args, **kwargs):
    return to_default_device(torch.full(*args, **kwargs))


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
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def get_fastest_index(indices: dict[IndexExpr, IndexSequence]):
    """
    This function takes in indices of a Node, extract their sizes
    into a list, and then try do an argmax on it. In the case where
    there are multipled max_vals we pick the fastest/most minor one.
    """

    index_sizes = [subs_idxc(i.size) for i in indices.values()]
    # Find the maximum value
    max_size = max(index_sizes)
    # Find the fastest/most minor index of the maximum value.
    return max(i for i, size in enumerate(index_sizes) if size == max_size)


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


def print_graph(graph: fx.Graph):
    """
    Pretty-print the graph containing this node.
    """
    graph_str = str(graph)
    graph_str = graph_str.replace(
        "iree.turbine.kernel.lang.kernel_buffer.KernelBufferMeta.new_subtype.<locals>.SubType",
        "",
    )
    graph_str = graph_str.replace("target=iree.turbine.kernel.ops.wave_ops.", "")
    graph_str = graph_str.replace("call_function", "")
    print(graph_str)


def is_reduction_subgraph(graph: fx.Graph):
    """
    Check that graph is a subgraph that is owned by ReductionOp.
    """
    if not hasattr(graph, "parent_op"):
        return False
    return isinstance(get_custom(graph.parent_op), Reduction)


def initialize_iter_args(trace: CapturedTrace) -> None:
    """
    Initializes the IterArgs in each reduction with an index
    based on their location in the graph.

    """
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Reduction))
    for reduction in reductions:
        reduction_graph = trace.get_subgraph(get_custom(reduction).subgraph_name)
        count = 0
        for node in reduction_graph.nodes:
            custom = get_custom(node)
            if isinstance(custom, IterArg):
                custom.iter_idx = count
                count += 1


def partial(func, *args, **kwargs):
    """functools.partial but with function attributes copied to the partial function."""
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


TORCH_DTYPE_TO_WAVE = {
    torch.bfloat16: tkl.bf16,
    torch.float8_e5m2: tkl.f8e5m2,
    torch.float8_e5m2fnuz: tkl.f8e5m2fnuz,
    torch.float8_e4m3fn: tkl.f8e4m3fn,
    torch.torch.float8_e4m3fnuz: tkl.f8e4m3fnuz,
    torch.float16: tkl.f16,
    torch.float32: tkl.f32,
    torch.float64: tkl.f64,
    torch.int16: tkl.i16,
    torch.int32: tkl.i32,
    torch.int64: tkl.i64,
    torch.bool: tkl.bool,
}


def torch_dtype_to_wave(torch_dtype: torch.dtype) -> Any:
    try:
        return TORCH_DTYPE_TO_WAVE[torch_dtype]
    except KeyError:
        raise ValueError(f"Unable to map torch dtype {torch_dtype} to Wave.")


def is_shared_write(node: CustomOp) -> bool:
    return (
        isinstance(node, Write)
        and subs_idxc(node.memory_type.address_space) == SHARED_ADDRESS_SPACE
    )


def is_shared_read(node: CustomOp) -> bool:
    return (
        isinstance(node, Read)
        and subs_idxc(node.memory_type.address_space) == SHARED_ADDRESS_SPACE
    )


def is_gather(custom: CustomOp) -> bool:
    if not isinstance(custom, Read):
        return False
    assert custom.index, f"Read node {custom} does not have an index."
    return any(
        custom.index[x].size > 1
        for x in custom.memory_type.symbolic_shape[:-1]
        if x in custom.index
    )


def print_live_tensors():
    """
    Print all alive torch tensors in program.

    Use for debugging memory leaks.
    """
    import gc

    gc.collect()

    print("------ live tensors ---------")
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(hex(id(obj)), type(obj), obj.size())
        except:
            pass
    print("-----------------------------")


def remove_files_with_extension(directory, extension):
    pattern = os.path.join(directory, "*" + extension)
    files_to_remove = glob.glob(pattern)

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
