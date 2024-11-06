# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import torch.fx as fx
from typing import Any, TypeAlias
from functools import partial

from .constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from ..ops.wave_ops import *
from .._support.indexing import IndexingContext, IndexSequence
from ...support.logging import get_logger
from .._support.tracing import CapturedTrace
from .utils import (
    get_mma_dimensional_mapping,
    specialize_index_sequence,
    get_hardware_constraint,
    get_workgroup_constraints,
)
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.expansion")
# This represents a mapping of a node + indexing + res_idx(output index for op with multiple results)
# of node into the dimensions to the corresponding expanded node in these specific dimensions.
# An example for a record in this map is (read_0_0_0, ((M,0),(N,0),(K,1), 0) -> read_0_0_1.
ExpandedNodeMap: TypeAlias = dict[
    tuple[CustomOp, tuple[tuple[IndexSymbol, int], int, ...]], CustomOp
]


def already_expanded_iter_arg(node: CustomOp, dims: dict[IndexSymbol, int]) -> bool:
    return (
        hasattr(node.fx_node, "expanded_dims")
        and isinstance(node, IterArg)
        and (
            filter_and_zero_unselected_dims(dims, node.indexing_dims)
            == node.fx_node.expanded_dims  # type: ignore
        )
    )


def expansion_needed(
    dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
) -> bool:
    """Check if any of the dimensions in the selection are non-zero."""
    return any(dim[1] != 0 and dim[0] in selection for dim in dims.items())


def filter_and_zero_unselected_dims(
    dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
) -> dict[IndexSymbol, int]:
    """
    Filters dimensions based on selection and sets unselected dimensions' values to zero.
    """
    return {dim: val if dim in selection else 0 for dim, val in dims.items()}


def get_dim_combinations(
    all_dims: dict[IndexSymbol, int],
    selection: Sequence[IndexSymbol],
):
    """
    Returns all combinations of sizes for the selected dimensions.
    Other dimensions are clamped to 0.
    """
    adjusted_dimension_sizes = [
        list(range(all_dims[dim])) if dim in selection else [0] for dim in all_dims
    ]
    return itertools.product(*adjusted_dimension_sizes)


def get_indexed_dims(
    all_dims: dict[IndexSymbol, int], nodeOrDims: CustomOp | Sequence[IndexSymbol]
) -> tuple[tuple[IndexSymbol, int], ...]:
    """
    Generates a tuple of (key, value) pairs from the provided dimensions.
    If given a CustomOp instance, it uses its indexing_dims attribute.
    """
    if isinstance(nodeOrDims, CustomOp):
        nodeOrDims = nodeOrDims.indexing_dims
    # Flatten dims for node with multiple values or expanded Reduction.
    if all(isinstance(el, Sequence) for el in nodeOrDims):
        flattened_dims = list(itertools.chain.from_iterable(nodeOrDims))
        flatten_dims_set = dict.fromkeys(flattened_dims)
        nodeOrDims = list(flatten_dims_set)
    return tuple((key, all_dims[key]) for key in nodeOrDims if key in all_dims)


def get_last(node_list: fx.graph._node_list) -> fx.Node:  # type: ignore
    """Get the last element of the fx node_list structure"""
    return next(iter(reversed(node_list)))  # type: ignore


def is_expandable(arg: Any) -> bool:
    """Check if an argument is expandable."""
    if isinstance(arg, list):
        return all(is_expandable(a) for a in arg)
    # Placeholder nodes are only expanded if they are a reduction init arg
    if isinstance(arg, Placeholder) and not isinstance(arg, IterArg):
        return False
    return isinstance(arg, CustomOp)


def expand_graph(
    trace: CapturedTrace,
    constraints_or_scaling: Sequence[Constraint] | dict[IndexSymbol, int],
):
    """
    Create a graph that represents the expanded version of the wave function.
    The expansion is done in the dimensions specified by the constraints.
    """
    get_node_dim_scaling = partial(get_dim_scaling, constraints_or_scaling)

    # Start from the back and expand in the corresponding indexing dimensions of a node
    # Then proceed to the operands
    leaf_nodes: list[Type[CustomOp]] = [Write]

    # Some graphs may not have a write node, so we need to add the leaf nodes present in the
    # graph, excluding output nodes.
    all_fx_nodes_reversed = list(reversed(trace.get_root_graph().nodes))
    has_write = any(
        isinstance(get_custom(fx_node), Write) for fx_node in all_fx_nodes_reversed
    )
    if not has_write:
        for node in (get_custom(fx_node) for fx_node in all_fx_nodes_reversed):
            if isinstance(node, Output):
                continue
            leaf_nodes.append(node.__class__)
            break

    expansion_context: ExpandedNodeMap = {}
    for node in (get_custom(fx_node) for fx_node in all_fx_nodes_reversed):

        # Expansion begins at the leaf nodes
        if node.__class__ not in leaf_nodes:
            continue

        dim_scaling = get_node_dim_scaling(node)
        for dim_combination in get_dim_combinations(dim_scaling, node.indexing_dims):
            expand_dims = {
                dim: val for dim, val in zip(dim_scaling.keys(), dim_combination)
            }
            logger.debug(f"Starting expansion at leaf:{node} in dims:{expand_dims}")
            _expand_node(
                node,
                trace,
                expand_dims,
                dim_scaling,
                expansion_context,
                get_node_dim_scaling,
            )


def _expand_node(
    node: CustomOp | list[CustomOp],
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int = 0,
) -> CustomOp:
    """Expand a single node or list of nodes in specific dimensions and recursively proceed to its inputs."""
    if isinstance(node, list):
        expanded_nodes = []
        for elem in node:
            expanded_nodes.append(
                _expand_node(
                    elem,
                    trace,
                    dim_query,
                    get_node_dim_scaling(elem),
                    context,
                    get_node_dim_scaling,
                    res_idx,
                ).fx_node
            )
        return expanded_nodes
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dim_query, node), res_idx) in context:
        logger.debug(f"Already expanded node: {node} in {dim_query}")
        return context[(node, get_indexed_dims(dim_query, node), res_idx)]
    elif isinstance(node, MMA):
        # Handle expansion of MMA nodes whose reduction dim is not the same as the reduction
        # dim of the parent reduction op or when there is no parent reduction op.
        has_parent_op = hasattr(node.graph, "parent_op")
        reduction_axes_different = False
        if has_parent_op:
            reduction: Reduction = get_custom(node.graph.parent_op)
            reduction_axes_different = reduction.axis != node.reduction_dim
        parallel_dim_query = node.reduction_dim not in dim_query
        if (not has_parent_op or reduction_axes_different) and parallel_dim_query:
            return _expand_mma_reduction(
                node,
                trace,
                dim_query,
                dim_scaling,
                context,
                get_node_dim_scaling,
                res_idx,
            )
    elif isinstance(node, Reduction):
        return _expand_reduction(
            node, trace, dim_query, dim_scaling, context, get_node_dim_scaling, res_idx
        )
    elif isinstance(node, Getitem):
        res_idx = node.res_idx
    elif isinstance(node, GetResult) and not isinstance(node, Getitem):
        # The presence of a GetResult node indicates that the reduction has already
        # been expanded. Simply return the corresponding node.
        reduction = get_custom(node.value)
        return context[(reduction, get_indexed_dims(dim_query, reduction), res_idx)]
    elif isinstance(node, Allocate):
        # Allocate nodes are not expanded.
        return node

    # Filter out the dimensions that are not indexed by the node
    restricted_dims = filter_and_zero_unselected_dims(dim_query, node.indexing_dims)
    logger.debug(f"Expanding node: {node} in {restricted_dims}")

    # For iter args, we want to insert
    if not hasattr(_expand_node, "last_expanded_iter_arg"):
        _expand_node.last_expanded_iter_arg = None

    # Clone the node for the new expansion. The original node is reused for the
    # case of all dimensions being zero.
    if expansion_needed(restricted_dims, node.indexing_dims):
        new_node = node.copy(
            anchor=(
                _expand_node.last_expanded_iter_arg
                if isinstance(node, IterArg)
                else None
            )
        )
    else:
        new_node = node
        logger.debug(f"did not clone node: {node} in {restricted_dims}")

    if isinstance(node, IterArg):
        _expand_node.last_expanded_iter_arg = new_node.fx_node

    new_node.expanded_dims = restricted_dims
    new_node.fx_node.name = get_expanded_name(node, restricted_dims)

    # For reshapes, we need more explicit control over how the arguments are expanded.
    if isinstance(new_node, Reshape):
        _expand_reshape(
            new_node,
            trace,
            dim_query,
            dim_scaling,
            context,
            get_node_dim_scaling,
            res_idx,
        )
        context[(node, get_indexed_dims(restricted_dims, node), res_idx)] = new_node
        return new_node

    # Proceed with expansion of the arguments
    for i, arg in node.node_args.items():
        arg_list = arg
        unpack = lambda x: x
        if isinstance(arg, list):
            if not all(is_expandable(a) for a in arg):
                continue
        else:
            arg_list = [arg]
            unpack = lambda x: x[0]
            if not is_expandable(arg):
                continue

        new_args = []
        for subarg in arg_list:
            new_subarg = _expand_node(
                subarg,
                trace,
                restricted_dims,
                get_node_dim_scaling(subarg),
                context,
                get_node_dim_scaling,
                res_idx,
            )
            new_args.append(new_subarg.fx_node)
        new_node.update_arg(i, unpack(new_args))

    context[(node, get_indexed_dims(restricted_dims, node), res_idx)] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int = 0,
) -> CustomOp:
    """Expand a reduction in a specific dimension and recursively proceed to its inputs."""
    # Determine the dimensions to expand the reduction from the indexing of its users
    users = reduction.users
    expand_dims: list[IndexSymbol] = []
    for user in users:
        dim_scaling.update(get_node_dim_scaling(user))
        for indexing_dim in user.indexing_dims:
            if indexing_dim not in expand_dims:
                expand_dims.append(indexing_dim)
    logger.debug(f"expanding reduction in dims: {expand_dims}")

    # Get the output node of the reduction
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    output = get_custom(get_last(reduction_subgraph.nodes))
    if not isinstance(output, Output):
        raise ValueError(
            "fx.Graph malformed: The last node of a subgraph must be an output node"
        )

    new_output_args = []
    new_init_args = []
    for dim_vals in get_dim_combinations(dim_scaling, expand_dims):
        return_vals = output.return_vals[0]
        dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
        if not isinstance(return_vals, Sequence):
            return_vals = [return_vals]
        # Proceed with expansion inside the reduction
        for arg_idx, arg in enumerate(return_vals):
            arg = get_custom(arg)
            # Add GetResult nodes for the corresponding dimensions
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            # Usually we would rely on infer_types inside add_to_graph to figure out
            # the type of the new node. However, in this case, the logic to determine
            # the type requires the reduction node to have its init_args set, which has
            # not happened yet (it happens later). So instead, since we have access to
            # arg, we just set the type directly.
            new_node.add_to_graph(reduction.graph, arg.type)
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[
                (reduction, get_indexed_dims(dims, expand_dims), arg_idx)
            ] = new_node

            expanded_output = _expand_node(
                arg,
                trace,
                dims,
                get_node_dim_scaling(arg),
                context,
                get_node_dim_scaling,
                res_idx,
            )
            # If condition below is needed to skip over induction variable
            # who doesn't have all dims of ReductionOp. For example,
            # a reduction Op that has induction variables of types
            # (max, mma) -> [M], [M, N]
            # will have indexing dims of ([M, N]).
            # However, the 1st induction variable won't expand in N-dim
            # M:0, N:0 expand(max) -> max_0_0_0
            # M:0, N:1 expand(max) -> max_0_0_0
            # but will get added to the `new_output_args` without the if condition.

            # TODO: Handle expansion of induction variables with "non-complete" dims
            #       by checking on the indexing_dims on each induction variable.
            if expanded_output in new_output_args:
                continue
            new_output_args.append(expanded_output)

        # Proceed with expansion outside the reduction
        for init_arg in reduction.init_args:
            custom_init_arg = get_custom(init_arg)
            expanded_init_arg = _expand_node(
                custom_init_arg,
                trace,
                dims,
                get_node_dim_scaling(custom_init_arg),
                context,
                get_node_dim_scaling,
                res_idx,
            )
            # TODO: Handle expansion of induction variables with "non-complete" dims
            #       by checking on the indexing_dims on each induction variable.
            if expanded_init_arg in new_init_args:
                continue
            new_init_args.append(expanded_init_arg)

    # Update init_args and return values
    reduction.update_arg(
        "init_args", [new_init_arg.fx_node for new_init_arg in new_init_args]
    )
    output.update_arg("return_vals", [node.fx_node for node in new_output_args])
    _handle_reduction_dim(
        reduction,
        output,
        trace,
        dim_scaling,
        context,
        get_node_dim_scaling,
        res_idx,
    )
    # Even though we expanded the reduction in multiple dimensions, we only return
    # the node corresponding to the original query
    return context[(reduction, get_indexed_dims(dim_query, expand_dims), res_idx)]


def _expand_mma_reduction(
    mma: MMA,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int,
) -> CustomOp:
    """
    This function expands an MMA node along its reduction dimension. It is called
    P times where P is the product of all of its parallel dimensions. For each
    invocation, we expand the reduction dimension.

    We first compute the dim scaling along the reduction dimension and then append
    it to the dim query so that the expanded node and its arguments can use the
    expanded dim query with the appropriate value of the reduction dimension.

    Unlike the reduction expansion, where we can do a separate expansion for each iter_arg,
    here we only have a single MMA node to start with. So we keep track of it and re-use
    it for all the expansions. We also keep track of the accumulator value to be used as
    the accumulator for the first expansion along the reduction dimension.
    """

    logger.debug(f"Expanding MMA reduction: {mma} in dims: {dim_query}")
    expand_dims = set(mma.indexing_dims) - set([mma.reduction_dim])

    idxc = IndexingContext.current()
    for dim in mma.indexing_dims:
        if dim not in dim_scaling and mma.vector_shapes[dim] > 0:
            tile_size = idxc.get_static_value(dim)
            dim_scaling[dim] = tile_size // mma.vector_shapes[dim]

    # Store the original mma node and accumulator value for expansion.
    # When we begin expansion, we have a single mma node with the correct accumulator.
    # This node corresponds to the dim query with all 0s and for this we reuse the
    # original mma node. For all other queries, we create a new node.
    # So say we have parallel dimensions {M, K2} and reduction dimension {K1}.
    # For M = 0, K2 = 0, K1 = 0, we use the original mma node.
    # For M = 0, K2 = 0, K1 = 1, we create a new node.
    # Now, when it is time to expand along new parallel dimensions, we use the original node
    # For M = 0, K2 = 1, K1 = 0, we use the original mma node so that the last cloned node's
    # accumulator value is not modified.

    dim_query_dims = tuple(dim_query.keys())
    if not hasattr(_expand_mma_reduction, "acc"):
        _expand_mma_reduction.acc = {}
    if not hasattr(_expand_mma_reduction, "mma"):
        _expand_mma_reduction.mma = {}
    if (
        dim_query_dims not in _expand_mma_reduction.mma
        or _expand_mma_reduction.mma[dim_query_dims].graph != mma.graph
    ):
        _expand_mma_reduction.mma[dim_query_dims] = mma
        _expand_mma_reduction.acc[dim_query_dims] = mma.acc

    context_key = (
        _expand_mma_reduction.mma[dim_query_dims],
        get_indexed_dims(dim_query, expand_dims),
        res_idx,
    )

    user = _expand_mma_reduction.mma[dim_query_dims]
    for scale_idx in range(dim_scaling[mma.reduction_dim]):
        if isinstance(user, Output):
            continue

        dims = dim_query
        dims[mma.reduction_dim] = scale_idx
        # Temporarily replace the loop carried arg here to avoid
        # duplicated expansion. Otherwise we have the following situation:
        # Suppose we have:
        #   mma_0_0_0(..., acc_0_0_0)
        #   mma_0_0_1(..., mma_0_0_0)
        # Expanding mma_0_0_1 to mma_0_0_2 will trigger expansion of its arg
        # mma_0_0_0 in dims 0_0_2 as well, effectively duplicating the new node.
        # To avoid this we temporarily replace the use of it with a dummy
        # placeholder which will not trigger further expansion.
        index = user.get_node_arg_index(get_custom(user.acc))
        dummy = Placeholder("dummy").add_to_graph(user.graph)
        dummy.type = None

        saved_arg = user.node_args[index]
        user.update_arg(index, dummy)
        new_node = _expand_node(
            user,
            trace,
            dims,
            get_node_dim_scaling(user),
            context,
            get_node_dim_scaling,
        )

        # Update the new node accumulator with the user, except the first one.
        if scale_idx > 0:
            new_node.update_arg(index, user)
        else:
            new_node.update_arg(index, _expand_mma_reduction.acc[dim_query_dims])
        user.update_arg(index, saved_arg)
        user.graph.erase_node(dummy)
        user = new_node

    context[context_key] = new_node
    return new_node


def _expand_reshape(
    reshape: Reshape,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int,
) -> CustomOp:
    """
    When expanding a reshape, we have to expand the arguments of the reshape and then concatenate them together
    for the expanded node. Say we have a node with indexing dims = [M, N] with vector shapes m=8, n=2 and
    the reshape wants to map it to m=4, n=4. So we start by expanding the node
    node: {m = 0, n = 0}
        arg: {m = 0, n = 0}
        arg: {m = 0, n = 1}
    node: {m = 1, n = 0}
        arg: {m = 0, n = 0}
        arg: {m = 0, n = 1}
    node: {m = 2, n = 0}
        arg: {m = 1, n = 0}
        arg: {m = 1, n = 1}
    node: {m = 3, n = 0}
        arg: {m = 1, n = 0}
        arg: {m = 1, n = 1}
    ...
    In general,
    For the (m = i, n = j) expansion of the reshape node, we expand the arguments of the reshape node
    using the following recipe:
    - if m_src < m_dst, => we have a one to many mapping from source to destination
        so we expand the arguments along m = i // (m_dst / m_src) and we expand the argument only once.
    - if m_src > m_dst, => we have a many to one mapping from source to destination
        so we expand the arguments along m = i * (m_src / m_dst), ... and we expand the argument m_dst / m_src times.

    In situations where the argument has been expanded along the same dimension, we reuse the expanded node
    by making use of the context.
    """

    dim_combinations = {}
    for dim, value in dim_query.items():
        if dim not in reshape.target_vector_shape:
            continue
        if reshape.vector_shapes[dim] < reshape.target_vector_shape[dim]:
            scale_factor = (
                reshape.target_vector_shape[dim] // reshape.vector_shapes[dim]
            )
            dim_combinations[dim] = [value // scale_factor]
        else:
            scale_factor = (
                reshape.vector_shapes[dim] // reshape.target_vector_shape[dim]
            )
            begin = value * scale_factor
            dim_combinations[dim] = list(range(begin, begin + scale_factor))
    reshape_dim_combinations = list(itertools.product(*dim_combinations.values()))

    new_args = []
    for i, arg_dim_query in enumerate(reshape_dim_combinations):
        arg_dim_query = {
            dim: val for dim, val in zip(dim_combinations.keys(), arg_dim_query)
        }
        if isinstance(reshape.args, Sequence):
            custom_arg = get_custom(reshape.args[i])
        else:
            custom_arg = get_custom(reshape.args)
        new_node = _expand_node(
            custom_arg,
            trace,
            arg_dim_query,
            get_node_dim_scaling(custom_arg.fx_node),
            context,
            get_node_dim_scaling,
            res_idx,
        )
        new_args.append(new_node.fx_node)

    reshape.update_arg("args", new_args)


def get_expanded_name(node: CustomOp, dims: dict[IndexSymbol, int]) -> str:
    """Returns the name of a node with the dimensions appended."""

    separated = node.fx_node.name.split("_")
    node_name = separated[0]
    if isinstance(node, Read) or isinstance(node, Write):
        if get_custom(node.memory).type.address_space == SHARED_ADDRESS_SPACE:
            node_name = node_name + "_shared"
    # Special case for get_result op
    if node_name == "get":
        node_name = node_name + separated[1]
    for val in dims.values():
        node_name += f"_{val}"
    return node_name


def _contains(elem, container):
    if container is None:
        return False

    return elem in container


def get_dim_scaling(
    constraints: Sequence[Constraint], node: fx.Node
) -> dict[IndexSymbol, int]:
    """Get the number of expansions for the dimensions based on the constraints for a specific node."""
    dim_scaling: dict[IndexSymbol, int] = {}
    if node.vector_shapes is None:
        return dim_scaling

    hardware_constraints: list[HardwareConstraint] = [
        constraint
        for constraint in constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if len(hardware_constraints) != 1:
        raise ValueError("Exactly one hardware constraint must be provided")

    idxc = IndexingContext.current()
    for constraint in constraints:
        if isinstance(constraint, WorkgroupConstraint) or isinstance(
            constraint, TilingConstraint
        ):
            hw_cons = hardware_constraints[0]
            tile_size = idxc.get_static_value(constraint.tile_size)
            if constraint.dim not in node.vector_shapes:
                continue
            vector_size = node.vector_shapes[constraint.dim]

            # No dim scaling for dims with 0 vector size.
            if vector_size == 0:
                continue

            wave_count = 1
            if isinstance(constraint, WorkgroupConstraint):
                wave_count = hw_cons.waves_per_block[constraint.workgroup_dim]
            if tile_size is None or wave_count is None or vector_size is None:
                raise ValueError(
                    "Tile size, wave count and vector size must be statically known"
                )
            if (
                tile_size % wave_count != 0
                or (tile_size / wave_count) % vector_size != 0
            ):
                raise ValueError(
                    "Tile size must be divisible by wave count and vector size"
                )
            dim_scaling[constraint.dim] = tile_size // wave_count // vector_size

    return dim_scaling


def _expand_mma_tiled_reduction(
    mma: MMA,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int,
) -> CustomOp:
    latest_reduced_op = mma
    # The initial nodes are expanded in the first dimension, so we start from 1
    for scale_idx in range(1, dim_scaling[mma.reduction_dim]):
        dim_query[mma.reduction_dim] = scale_idx
        # Temporarily replace the loop carried arg here to avoid
        # duplicated expansion. Otherwise we have the following situation:
        # Suppose we have:
        #   mma_0_0_0(..., acc_0_0_0)
        #   mma_0_0_1(..., mma_0_0_0)
        # Expanding mma_0_0_1 to mma_0_0_2 will trigger expansion of its arg
        # mma_0_0_0 in dims 0_0_2 as well, effectively duplicating the new node.
        # To avoid this we temporarily replace the use of it with a dummy
        # placeholder which will not trigger further expansion.
        dummy = Placeholder("dummy").add_to_graph(latest_reduced_op.graph)
        dummy.type = None

        saved_acc = latest_reduced_op.acc
        latest_reduced_op.update_arg("acc", dummy)
        new_node = _expand_node(
            latest_reduced_op,
            trace,
            dim_query,
            dim_scaling,
            context,
            get_node_dim_scaling,
            res_idx,
        )

        # Node is always cloned; Hence, will never be equal to latest reduced op
        assert new_node != latest_reduced_op
        # Update MMA_{t} to accumulate on MMA_{t-1}, and then save
        # current MMA_{t} to outputs for use in next loop.
        latest_reduced_op.update_arg("acc", saved_acc)
        new_node.update_arg("acc", latest_reduced_op)
        latest_reduced_op.graph.erase_node(dummy)
        latest_reduced_op = new_node
    return latest_reduced_op


def _handle_reduction_dim(
    reduction: Reduction,
    output: Output,
    trace: CapturedTrace,
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    get_node_dim_scaling: Callable[[fx.Node], dict[IndexSymbol, int]],
    res_idx: int,
):
    # Rediscover iter args
    # TODO: Register iter args with the reduction initially so accessing them is easier
    iter_args: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)

    # TODO: Handle case where MMAs/ReduceOps do not have Output as direct consumer.
    def get_output_index(custom: CustomOp):
        output_users = [
            get_custom(user)
            for user in custom.fx_node.users
            if isinstance(get_custom(user), Output)
        ]
        if len(output_users) != 1:
            raise NotImplementedError(
                "NYI: Currently only handle direct and 1:1 MMA -> Output case."
            )
        return output_users[0].return_vals[0].index(custom.fx_node)

    # Collect MMA and ReduceOp who's reduction axis matches parent ReductionOp.
    reduction_root_ops = []
    for node in (get_custom(fx_node) for fx_node in reduction_subgraph.nodes):
        if isinstance(node, (MMA, ReduceOp)) and reduction.axis == node.reduction_dim:
            reduction_root_ops.append(node)

    new_outputs = list(reduction.outputs(trace.get_subgraph(reduction.subgraph_name)))
    # Users of the loop carried nodes will be duplicated
    for root_op in reduction_root_ops:
        dim_scaling = get_node_dim_scaling(root_op)
        dims = dict(root_op.fx_node.expanded_dims)
        latest_reduced_op = root_op
        op_output_index = get_output_index(root_op)
        if isinstance(root_op, MMA):
            latest_reduced_op = _expand_mma_tiled_reduction(
                root_op,
                trace,
                dims,
                dim_scaling,
                context,
                get_node_dim_scaling,
                res_idx,
            )
        elif isinstance(root_op, ReduceOp):
            original_src = latest_reduced_op.arg
            # The initial nodes are expanded in the first dimension, so we start from 1
            for scale_idx in range(1, dim_scaling[reduction.axis]):
                dims[root_op.reduction_dim] = scale_idx
                current_src = latest_reduced_op.arg
                if not isinstance(current_src, Sequence):
                    current_src = [current_src]
                expanded_src = _expand_node(
                    get_custom(original_src),
                    trace,
                    dims,
                    dim_scaling,
                    context,
                    get_node_dim_scaling,
                    res_idx,
                )
                current_src.append(expanded_src.fx_node)
                latest_reduced_op.update_arg("arg", current_src)
        new_outputs[op_output_index] = latest_reduced_op.fx_node
        init_dims = root_op.fx_node.expanded_dims
        context[
            (root_op, get_indexed_dims(init_dims, root_op), res_idx)
        ] = latest_reduced_op
    output.update_arg("return_vals", new_outputs)
