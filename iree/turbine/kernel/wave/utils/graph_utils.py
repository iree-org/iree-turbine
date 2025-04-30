# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_utils import subs_idxc
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import *
import iree.turbine.kernel.lang as tkl
from ...ops.wave_ops import (
    get_custom,
    Write,
    NestedRegionOp,
    Output,
    SetSymbol,
    SharedMemoryBarrier,
    ExtractSlice,
    GetResult,
    CustomOp,
    Iterate,
    Placeholder,
    Conditional,
    MMA,
    IterArg,
)
from ..._support.indexing import IndexSymbol
from typing import Callable, Sequence
import torch.fx as fx


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
        if isinstance(custom, Iterate):
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
                if parent_reduction.users:
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
        if reduction is None:
            reduction = custom.parent_op()
        iter_arg_idx = custom.iter_idx
        inputs.append(reduction.init_args[iter_arg_idx])
    elif isinstance(custom, GetResult):
        assert custom.value is not None, f"GetResult node {custom} has no value"
        reduction = get_custom(custom.value)
        assert isinstance(
            reduction, Iterate
        ), f"GetResult must be using a Reduction, but\n{custom}\nis using\n{reduction}"
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
    elif isinstance(custom, Iterate):
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


def replace_uses_in(users: dict[fx.Node, list[CustomOp]], old: CustomOp, new: fx.Node):
    """
    Replace all uses of `old` with `new` in the list of users.
    """
    for user in users[old]:
        for i, arg in enumerate(user.fx_node.args):
            if arg == old.fx_node:
                user.update_arg(i, new)


def is_reduction_subgraph(graph: fx.Graph):
    """
    Check that graph is a subgraph that is owned by ReductionOp.
    """
    if not hasattr(graph, "parent_op"):
        return False
    return isinstance(get_custom(graph.parent_op), Iterate)


def initialize_iter_args(trace: CapturedTrace) -> None:
    """
    Initializes the IterArgs in each reduction with an index
    based on their location in the graph.

    """
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Iterate))
    for reduction in reductions:
        reduction_graph = trace.get_subgraph(get_custom(reduction).subgraph_name)
        count = 0
        for node in reduction_graph.nodes:
            custom = get_custom(node)
            if isinstance(custom, IterArg):
                custom.iter_idx = count
                count += 1
