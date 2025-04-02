# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..._support.tracing import CapturedTrace
from typing import Sequence, Type, Any
from ..constraints import (
    Constraint,
)
from ...ops.wave_ops import (
    Allocate,
    CustomOp,
    get_custom,
    Output,
    Write,
    Reduction,
    ReduceOp,
    IterArg,
    Reshape,
    GetResult,
    MMA,
    SetSymbol,
    ApplyExpr,
    Broadcast,
)
from ..._support.indexing import IndexingContext, IndexSymbol
import itertools
from torch import fx
from dataclasses import dataclass
from .expansion_utils import (
    get_dim_scaling,
    flatten_list,
    get_indexed_dims,
    is_expandable,
    get_expanded_name,
    compute_strides,
    ExpansionMetadata,
    get_reshape_dim_queries,
    remove_original_nodes,
    remove_unused_registers,
)
from ..utils.graph_utils import (
    get_users,
    get_inputs,
)
from ....support.logging import get_logger
from copy import deepcopy
import math

logger = get_logger("turbine.wave.expansion")


@dataclass(frozen=True)
class ExpansionInfo:
    """
    Key used to store and look up nodes during expansion.
    """

    node: CustomOp
    indexed_dims: tuple[tuple[IndexSymbol, int], ...]

    def __repr__(self):
        return f"ExpansionInfo({self.node.fx_node}, {self.indexed_dims})"


class ReductionInfo:
    """
    Contains fixup information for a reduction node.
    """

    def __init__(self, reduction: Reduction):
        self.reduction = reduction
        self.outputs: dict[int, ExpansionInfo] = {}
        self.init_args: dict[int, ExpansionInfo] = {}
        self.get_results: dict[int, CustomOp] = {}

    def __repr__(self):
        get_results = {i: c.fx_node for i, c in self.get_results.items()}
        return (
            f"ReductionInfo({self.reduction.fx_node},"
            f"outputs={self.outputs},"
            f" init_args={self.init_args},"
            f" get_results={get_results}"
        )


class ExpansionContext:
    """
    Context used to store information during expansion.
    """

    def __init__(self):
        self.expansion_context: dict[ExpansionInfo, CustomOp] = {}
        # Additional operator specific information.
        self.reduction_context: dict[Reduction, ReductionInfo] = {}
        self.mma_connections: list[tuple[MMA, MMA]] = []
        self.mma_nodes: list[tuple[MMA]] = []

    def __getitem__(self, key: ExpansionInfo):
        return self.expansion_context[key]

    def __contains__(self, key: ExpansionInfo):
        return key in self.expansion_context

    def __setitem__(self, key: ExpansionInfo, value: CustomOp):
        self.expansion_context[key] = value


def get_dim_combinations(
    node: CustomOp,
    constraints: Sequence[Constraint],
):
    """
    Returns all combinations of sizes for the selected dimensions.
    Other dimensions are clamped to 0. A dictionary is return where
    the keys are the dimensions and the values are the combination.
    """
    dim_scaling = get_dim_scaling(constraints, node)
    adjusted_dimension_sizes = [
        list(range(dim_scaling[dim])) if dim in node.indexing_dims else [0]
        for dim in dim_scaling
    ]
    dim_combinations = itertools.product(*adjusted_dimension_sizes)
    return [
        {dim: val for dim, val in zip(dim_scaling.keys(), dim_combination)}
        for dim_combination in dim_combinations
    ]


def filter_expandable_args(args: list[Any]) -> list[Any]:
    """
    Filter out the arguments that can be expanded. These are the arguments
    that are of type CustomOp.
    """
    filtered_args = []
    for arg in args:
        arg_list = arg
        if isinstance(arg, Sequence):
            if not all(is_expandable(arg) for arg in arg):
                continue
        else:
            if not is_expandable(arg):
                continue
            arg_list = [arg]
        filtered_args.append(arg_list)
    return flatten_list(filtered_args)


def filter_and_zero_unselected_dims(
    dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
) -> dict[IndexSymbol, int]:
    """
    Filters dimensions based on selection and sets unselected dimensions' values to zero.
    """
    return {dim: val if dim in selection else 0 for dim, val in dims.items()}


def compute_result_index(
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    node: fx.Node,
    outputs: list[fx.Node],
):
    """
    Compute the result index for a reduction node based on the dim
    query and dim scaling.

    Say we have a reduction with output:
    (max, sum, mma)

    and say that the indexing dims are:
    max -> (M)
    sum -> (M)
    mma -> (M, N)

    and if the dim_scaling is:
    M -> 2
    N -> 2

    then we know that there will be a total of 8 results and
    we can arrange them as follows:
    (max_M:0, max_M:1, sum_M:0, sum_M:1, mma_M:0_N:0, mma_M:0_N:1, mma_M:1_N:0, mma_M:1_N:1)

    This means that each of these results has a predefined index that we can compute
    based on the dim_scaling and dim_query.

    The formula for inputs in general is:

    # Global offset.
    global_index = 0
    for i in range(index(input)):
        global_index += product(dim_scaling[d] for d in node[i].indexing_dims)

    # Local offset.
    local_index += dim_query * dim_strides

    """
    input_index = outputs.index(node)
    get_shape = lambda x: get_custom(x).type.symbolic_shape
    result_index = sum(
        math.prod(dim_scaling[d] for d in get_shape(outputs[i]) if d in dim_scaling)
        for i in range(input_index)
    )
    node_shape = get_shape(node)
    restricted_dim_scaling = {k: v for k, v in dim_scaling.items() if k in node_shape}
    restricted_dim_query = {k: v for k, v in dim_query.items() if k in node_shape}
    result_index += sum(
        x * y
        for x, y in zip(
            compute_strides(restricted_dim_scaling), restricted_dim_query.values()
        )
    )
    return result_index


def to_tuple(d: dict[IndexSymbol, int]) -> tuple[int, ...]:
    return tuple([(k, v) for k, v in d.items()])


def to_dict(t: tuple[int, ...]) -> dict[IndexSymbol, int]:
    return {k: v for k, v in t}


def handle_reduction_entry(
    reduction: Reduction,
    inputs: list[CustomOp],
    new_node: CustomOp,
    node: CustomOp,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    expansion_context: ExpansionContext,
):
    reduction_context = expansion_context.reduction_context
    if isinstance(new_node, GetResult):
        assert len(inputs) == 1, f"Expected one input, got {inputs}"
        outputs = reduction.outputs(inputs[0].graph)
        if not isinstance(outputs, Sequence):
            outputs = [outputs]
        if reduction not in reduction_context:
            reduction_context[reduction] = ReductionInfo(reduction)
        result_index = compute_result_index(dim_query, dim_scaling, inputs[0], outputs)
        custom = get_custom(inputs[0])
        key = ExpansionInfo(custom, get_indexed_dims(dim_query, custom))
        reduction_info = reduction_context[reduction]
        assert (
            result_index not in reduction_info.outputs
            and result_index not in reduction_info.get_results
        ), f"{result_index=} has already been computed for {reduction_info}"
        reduction_info.outputs[result_index] = key
        reduction_info.get_results[result_index] = new_node


def handle_reduction_exit(
    reduction: Reduction,
    inputs: list[CustomOp],
    new_node: CustomOp,
    node: CustomOp,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    expansion_context: ExpansionContext,
):
    # If we are an iter arg, then we are exiting a reduction.
    reduction_context = expansion_context.reduction_context
    if isinstance(new_node, IterArg):
        assert len(inputs) == 1, f"Expected one input, got {inputs}"
        reduction = new_node.parent_op()
        result_index = compute_result_index(
            dim_query, dim_scaling, inputs[0], reduction.init_args
        )
        assert reduction in reduction_context, f"Reduction not found: {reduction}"
        new_node.iter_idx = result_index
        custom = get_custom(inputs[0])
        key = ExpansionInfo(custom, get_indexed_dims(dim_query, custom))
        reduction_context[reduction].init_args[result_index] = key


def concatenate_outputs(
    user: CustomOp,
    new_user: CustomOp,
    node: CustomOp,
    new_node: CustomOp,
    i: int,
    metadata: ExpansionMetadata,
):
    reshape_check = isinstance(new_user, Reshape)
    reduce_check = isinstance(new_user, ReduceOp) and i == 0
    if reshape_check or reduce_check:
        if metadata.query_index == 0:
            new_node = [new_node.fx_node]
        else:
            assert (
                metadata.query_index > 0
            ), f"Expected query index > 0, got {metadata.query_index}"
            new_node = [x.fx_node for x in new_user.node_args[i]] + [new_node.fx_node]
        return new_node
    return replace_node(user, new_user, node, new_node, i)


def replace_node(
    user: CustomOp, new_user: CustomOp, node: CustomOp, new_node: CustomOp, i: int
):
    # If we are updating a single value in a sequence, then we need to
    # insert the new node at the correct location.
    if isinstance(user.node_args[i], Sequence) and not isinstance(new_node, Sequence):
        new_node = [
            x.fx_node if x != node else new_node.fx_node for x in new_user.node_args[i]
        ]
    return new_node


def update_users(
    node: CustomOp,
    new_node: CustomOp,
    metadata: ExpansionMetadata,
    expansion_context: ExpansionContext,
):
    users, _ = get_users(node.fx_node, None)
    for user in users:
        user = get_custom(user)
        dim_query = metadata.dim_query
        # For reshapes and reduces, multiple users can share the same source.
        if isinstance(user, (Reshape, ReduceOp)):
            if not metadata.source_dim_query:
                continue
            dim_query = metadata.source_dim_query
        key = ExpansionInfo(user, get_indexed_dims(dim_query, user))
        if key in expansion_context:
            new_user = expansion_context[key]
            if not new_user.node_args:
                continue
            indices = user.get_node_arg_index(node)
            if indices is None:
                continue
            if not isinstance(indices, Sequence):
                indices = [indices]
            for i in indices:
                # Check if an update is required.
                if isinstance(new_user.node_args[i], Sequence):
                    # If node is already in the list, then we don't need to update.
                    if any(x == new_node for x in new_user.node_args[i]):
                        continue
                else:
                    if new_user.node_args[i] == new_node:
                        continue
                new_arg = concatenate_outputs(
                    user, new_user, node, new_node, i, metadata
                )
                new_user.update_arg(i, new_arg)


def add_to_outputs(node: CustomOp, new_node: CustomOp):
    """
    Add the new node to the outputs of the node at the correct index.
    """
    output = [x for x in node.users if isinstance(get_custom(x), Output)]
    if not output:
        return
    output = get_custom(output[0])
    users, _ = get_users(new_node.fx_node, None)
    get_result = [x for x in users if isinstance(get_custom(x), GetResult)]
    assert len(get_result) == 1, f"Expected one GetResult, got {get_result}"
    result_index = get_result[0].result_index
    if len(output.return_vals[0]) < result_index:
        new_return_vals = output.return_vals[0] + [None] * (
            result_index - len(output.return_vals[0]) + 1
        )
    new_return_vals[result_index] = new_node
    output.return_vals = [new_return_vals]


def get_node(
    dim_query: dict[IndexSymbol, int],
    node: CustomOp,
    expansion_context: ExpansionContext,
):
    key = ExpansionInfo(node, get_indexed_dims(dim_query, node))
    assert key in expansion_context, f"Key not found: {key}"
    return expansion_context[key]


def get_mma_reduction_count(arg: MMA, dim_scaling: dict[IndexSymbol, int]) -> int:
    if arg.reduction_dim in dim_scaling:
        reduction_count = dim_scaling[arg.reduction_dim]
    else:
        idxc = IndexingContext.current()
        tile_size = idxc.get_static_value(arg.reduction_dim)
        assert tile_size, f"Dimension not known : {arg.reduction_dim}"
        reduction_count = max(tile_size // arg.vector_shapes[arg.reduction_dim], 1)
    return reduction_count


def add_get_results(trace: CapturedTrace):
    reductions = trace.walk(lambda x: isinstance(get_custom(x), Reduction))
    for reduction in reductions:
        reduction = get_custom(reduction)
        if len(reduction.init_args) == 1:
            reduction.graph.inserting_after(reduction.fx_node)
            get_result = get_custom(
                GetResult(reduction.fx_node, 0).add_to_graph(reduction.graph)
            )
            reduction.replace_all_uses_with_except(get_result, [get_result])


def populate_inputs(
    node: CustomOp,
    inputs: list[fx.Node],
    metadata: ExpansionMetadata,
    dim_scaling: dict[IndexSymbol, int],
    nodes_to_expand: list[tuple[CustomOp, dict[IndexSymbol, int]]],
    expansion_context: ExpansionContext,
):
    expandable_args = filter_expandable_args([get_custom(x) for x in inputs])
    new_nodes_to_expand = []

    if isinstance(node, (Reshape, ReduceOp)):
        match node:
            case Reshape():
                dim_queries = get_reshape_dim_queries(
                    node, metadata, dim_scaling, new_nodes_to_expand
                )
            case ReduceOp():
                try:
                    reduction_count = dim_scaling[node.reduction_dim]
                except KeyError as e:
                    raise RuntimeError(
                        f"Reduction dimension {node.reduction_dim} is not in {dim_scaling} for ReduceOp {node}"
                    )
                dim_queries = []
                for i in range(reduction_count):
                    dim_query = deepcopy(metadata.dim_query)
                    dim_query[node.reduction_dim] = i
                    dim_queries.append(dim_query)

        count = 0
        for i, arg in enumerate(expandable_args):
            # For the init arg of the reduce op, if it exists, we expand only once.
            if isinstance(node, ReduceOp) and node.init and arg == node.init:
                nodes_to_expand.append((arg, metadata))
                continue
            for j, query in enumerate(dim_queries):
                new_metadata = deepcopy(metadata)
                new_metadata.dim_query = query
                new_metadata.source_dim_query = metadata.dim_query
                new_metadata.num_queries = len(dim_queries)
                new_metadata.query_index = count
                new_nodes_to_expand.append((arg, new_metadata))
                count += 1
        nodes_to_expand.extend(new_nodes_to_expand)
        return nodes_to_expand

    for arg in expandable_args:
        match arg:
            case MMA():
                reduction_count = get_mma_reduction_count(arg, dim_scaling)
                for i in range(reduction_count):
                    mma_metadata = deepcopy(metadata)
                    mma_metadata.dim_query[arg.reduction_dim] = i
                    if i == reduction_count - 1:
                        mma_metadata.last_mma_node = True
                    new_nodes_to_expand.append((arg, mma_metadata))
                continue
            case Allocate() | SetSymbol():
                alloc_metadata = deepcopy(metadata)
                alloc_metadata.do_not_expand = True
                new_nodes_to_expand.append((arg, alloc_metadata))
                continue

        new_nodes_to_expand.append((arg, metadata))

    nodes_to_expand.extend(new_nodes_to_expand)

    return nodes_to_expand


def store_fixup_data(
    node: CustomOp,
    new_node: CustomOp,
    expanded_dims: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    metadata: ExpansionMetadata,
    expansion_context: ExpansionContext,
):
    """
    Keep track of which MMA nodes need to be connected and replaced
    for the fixup phase.
    """
    match node:
        case MMA():
            try:
                if expanded_dims[node.reduction_dim] == 0:
                    return
            except KeyError as e:
                raise RuntimeError(
                    f"Reduction dim {node.reduction_dim} not in expanded dims {expanded_dims} for {node}"
                )

            def get_dim_query(new_v: int):
                dims = {
                    k: v if k != node.reduction_dim else new_v
                    for k, v in expanded_dims.items()
                }
                return dims

            # Update accumulator.
            last_dim_query = get_dim_query(expanded_dims[node.reduction_dim] - 1)
            last_node = get_node(last_dim_query, node, expansion_context)
            expansion_context.mma_connections.append((new_node, last_node))

            # Keep track of fixup nodes.
            if metadata.last_mma_node:
                first_node = get_node(get_dim_query(0), node, expansion_context)
                second_node = get_node(get_dim_query(1), node, expansion_context)
                expansion_context.mma_nodes.append((first_node, new_node, second_node))


def expand_node(
    node: CustomOp,
    dim_scaling: dict[IndexSymbol, int],
    nodes_to_expand: list[tuple[CustomOp, dict[IndexSymbol, int], int]],
    metadata: ExpansionMetadata,
    expansion_context: ExpansionContext,
):
    """
    When we expand a node, we clone it and add its arguments to the
    list of nodes to be expanded.
    """

    # Filter out the dimensions that are not selected in the query.
    expanded_dims = filter_and_zero_unselected_dims(
        metadata.dim_query, node.indexing_dims
    )

    # Check if the node has already been expanded, if so return early.
    indexed_dims = get_indexed_dims(expanded_dims, node)

    key = ExpansionInfo(node, indexed_dims)
    if key in expansion_context:
        update_users(node, expansion_context[key], metadata, expansion_context)
        return nodes_to_expand

    if metadata.do_not_expand:
        return nodes_to_expand

    # Make a copy of the node, adjust its name and set any metadata.
    new_node = node.copy(anchor=(node.fx_node.prev))
    new_node.fx_node.name = get_expanded_name(node, metadata.dim_query)
    new_node.expanded_dims = expanded_dims

    # Add new node to expansion context.
    expansion_context[key] = new_node

    # Store information needed for the fixup phase.
    store_fixup_data(
        node,
        new_node,
        expanded_dims,
        dim_scaling,
        metadata,
        expansion_context,
    )

    # Check for any expanded users and update their arguments.
    update_users(node, new_node, metadata, expansion_context)

    # Add expandable inputs to the list of nodes to expand.
    inputs, reduction = get_inputs(node.fx_node, None)

    handle_reduction_entry(
        reduction,
        inputs,
        new_node,
        node,
        metadata.dim_query,
        dim_scaling,
        expansion_context,
    )
    handle_reduction_exit(
        reduction,
        inputs,
        new_node,
        node,
        metadata.dim_query,
        dim_scaling,
        expansion_context,
    )

    nodes_to_expand = populate_inputs(
        node, inputs, metadata, dim_scaling, nodes_to_expand, expansion_context
    )
    return nodes_to_expand


def dfs(
    node: CustomOp,
    dim_query: dict[IndexSymbol, int],
    constraints: Sequence[Constraint],
    expansion_context: ExpansionContext,
):
    """
    Perform a depth-first search on the graph starting at the given node
    for the given dimension combination.
    """

    visited = set()
    nodes_to_expand = [(node, ExpansionMetadata(dim_query))]
    while nodes_to_expand:
        node, metadata = nodes_to_expand.pop(0)
        if (node, metadata) in visited:
            continue
        visited.add((node, metadata))
        dim_scaling = get_dim_scaling(constraints, node)
        nodes_to_expand = expand_node(
            node, dim_scaling, nodes_to_expand, metadata, expansion_context
        )


def get_last(node_list: fx.graph._node_list) -> fx.Node:  # type: ignore
    """Get the last element of the fx node_list structure"""
    return next(iter(reversed(node_list)))  # type: ignore


def fixup_mma_nodes(trace: CapturedTrace, expansion_context: ExpansionContext):
    # Chain MMA connections
    for current, last in expansion_context.mma_connections:
        current.update_arg(2, last)
    # Use the last MMA node instead of the first one.
    for first, second, exclude in expansion_context.mma_nodes:
        first.replace_all_uses_with_except(second, [exclude])


def get_mma_indexed_dims(
    mma: MMA,
    original_indexed_dims: tuple[tuple[IndexSymbol, int]],
    expansion_context: ExpansionContext,
):
    dim = mma.reduction_dim
    indexed_dims = None
    max_reduction_dim = -1
    original_indexed_dims_dict = dict(original_indexed_dims)
    for key in expansion_context.expansion_context.keys():
        indexed_dims_dict = dict(key.indexed_dims)
        if any(
            dim not in indexed_dims_dict
            or indexed_dims_dict[dim] != original_indexed_dims_dict[dim]
            for dim in original_indexed_dims_dict
        ):
            continue
        if key.node == mma and indexed_dims_dict[dim] > max_reduction_dim:
            indexed_dims = key.indexed_dims
            max_reduction_dim = indexed_dims_dict[dim]
    return indexed_dims


def fixup_reduction_nodes(
    trace: CapturedTrace,
    expansion_context: ExpansionContext,
):
    """
    This function fixes up the reduction nodes by updating the outputs,
    init_args and get_results of the reduction nodes. It also removes the
    original nodes from the graph.

    In situations where we have multiple reductions, and the outputs of a
    reduction are used as inputs to another reduction, we need to ensure
    the fixup is done in the correct order, specifically from the last
    reduction to the first reduction since that is the order in
    which expansion proceeds.
    """
    reduction_context = expansion_context.reduction_context
    reduction_nodes = trace.walk(lambda x: isinstance(get_custom(x), Reduction))
    for reduction in reversed(reduction_nodes):
        reduction = get_custom(reduction)
        reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
        output = get_custom(get_last(reduction_subgraph.nodes))
        return_vals = output.return_vals[0]
        if isinstance(return_vals, Sequence):
            return_vals = [get_custom(x) for x in output.return_vals[0]]
        else:
            return_vals = [get_custom(return_vals)]
        reduction_info = reduction_context[reduction]
        sorted_keys = dict(sorted(reduction_info.outputs.items(), key=lambda x: x[0]))
        new_outputs = []
        for key in sorted_keys.values():
            if key not in expansion_context and isinstance(key.node, MMA):
                key = ExpansionInfo(
                    key.node,
                    get_mma_indexed_dims(key.node, key.indexed_dims, expansion_context),
                )
                assert key in expansion_context, f"Key not found: {key}"
            new_outputs.append(expansion_context[key].fx_node)
        output.update_arg("return_vals", new_outputs)

        sorted_keys = dict(sorted(reduction_info.init_args.items(), key=lambda x: x[0]))
        new_init_args = []
        for key in sorted_keys.values():
            new_init_args.append(expansion_context[key].fx_node)
        reduction.update_arg("init_args", new_init_args)

        for result_index, get_item in reduction_info.get_results.items():
            get_item.graph.inserting_before(get_item.fx_node)
            get_result = GetResult(get_item.value, result_index).add_to_graph(
                get_item.graph, get_item.type
            )
            get_result.name = get_item.fx_node.name
            get_item.replace_all_uses_with(get_custom(get_result))
            get_item.erase()

        remove_original_nodes(return_vals)


def is_leaf_node(node):
    custom = get_custom(node)
    return isinstance(custom, Write) or (
        isinstance(custom, GetResult) and not custom.users
    )


def expand_graph(
    trace: CapturedTrace,
    constraints: Sequence[Constraint],
):
    """
    Create a graph that represents the expanded version of the wave function.
    The constraints are used to determine how the graph should be expanded.
    The expansion does a DFS starting at the leaf nodes and expanding them
    to the root of the graph.
    """

    leaf_ops = [get_custom(node) for node in reversed(trace.walk(is_leaf_node))]
    if not leaf_ops:
        final_op = get_custom(trace.get_root_graph()._root.prev)
        leaf_ops.append(final_op)
        logger.warning(
            f"No leaf operations found in kernel. Using final operation {final_op}"
        )
    expansion_context = ExpansionContext()
    for custom in leaf_ops:
        for dim_combination in get_dim_combinations(custom, constraints):
            dfs(
                custom,
                dim_combination,
                constraints,
                expansion_context,
            )

    # Fixup all reduction nodes.
    fixup_reduction_nodes(trace, expansion_context)
    # Fixup all mma nodes.
    fixup_mma_nodes(trace, expansion_context)
    # Remove original nodes in root graph.
    remove_original_nodes(leaf_ops)
    remove_unused_registers(trace)
