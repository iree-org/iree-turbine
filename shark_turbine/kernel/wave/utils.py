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
from typing import Optional, Callable, Any, List, Tuple
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
    IterArg,
)
from .constraints import (
    Constraint,
    WorkgroupConstraint,
    HardwareConstraint,
    TilingConstraint,
)
import torch.fx as fx
import shark_turbine.kernel.lang as tkl

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


def get_mma_dimensional_mapping(
    trace: CapturedTrace,
) -> tuple[dict[IndexSymbol, int], dict[IndexSymbol, list[fx.Node]]]:
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
    mma_nodes = trace.walk(is_mma)
    for node in mma_nodes:
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape
        k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        mapping[m] = 0
        mapping[n] = 1
        mapping[k] = 2

    return mapping, capture_mma_slices([get_custom(x) for x in mma_nodes])


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
        if isinstance(input, torch.Tensor):
            input_cpu = input.cpu().contiguous()
            device_array = rt.asdevicearray(device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)
        elif isinstance(input, int):
            arg_list.push_int(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

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
        custom = get_custom(user)
        if isinstance(custom, Reduction):
            # Map init arg to iter arg
            reduction = custom
            init_arg_idx = custom.init_args.index(node)
            users.append(custom.iter_args[init_arg_idx])
            continue
        if isinstance(custom, Output) and reduction:
            # Map output to get result
            return_vals = custom.return_vals[0]
            get_results = sorted(
                [x for x in reduction.users if isinstance(get_custom(x), GetResult)],
                lambda x: get_custom(x).res_idx,
            )
            if isinstance(return_vals, list):
                output_idx = return_vals.index(node)
                users.append(get_results[output_idx])
            else:
                users.append(get_results[0])
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
        assert isinstance(
            get_custom(reduction), Reduction
        ), "GetResult must be used by a Reduction"
        # Map get result to output
        reduction_subgraph = reduction.graph.subgraphs[reduction.subgraph_name]
        inputs.append(reduction.outputs(reduction_subgraph)[custom.res_idx])
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
    Run BFS on the graph to capture the forward slice of a node.
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
    node: fx.Node, filter_fn: Callable[[fx.node], bool] = lambda x: True
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


def capture_mma_slices(mma_nodes: list[MMA]) -> dict[IndexSymbol, list[fx.Node]]:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    """
    mma_slices = {x: [] for x in [MMA_LHS, MMA_RHS, MMA_ACC]}
    for mma in mma_nodes:
        mma_slices[MMA_LHS] += capture_backward_slice(mma.lhs)
        mma_slices[MMA_RHS] += capture_backward_slice(mma.rhs)
        mma_slices[MMA_ACC] += capture_forward_slice(mma.acc)
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


def replace_uses_in(users: dict[fx.Node, list[CustomOp]], old: CustomOp, new: fx.Node):
    """
    Replace all uses of `old` with `new` in the list of users.
    """
    for user in users[old]:
        for i, arg in enumerate(user.fx_node.args):
            if arg == old.fx_node:
                user.update_arg(i, new)
