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
    if isinstance(constraints_or_scaling, dict):
        dim_scaling = constraints_or_scaling
    else:
        dim_scaling = get_dim_scaling(constraints_or_scaling)

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
            )


def _expand_node(
    node: CustomOp | list[CustomOp],
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
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
                    dim_scaling,
                    context,
                    res_idx,
                ).fx_node
            )
        return expanded_nodes
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dim_query, node), res_idx) in context:
        logger.debug(f"Already expanded node: {node} in {dim_query}")
        return context[(node, get_indexed_dims(dim_query, node), res_idx)]
    elif isinstance(node, Reduction):
        return _expand_reduction(node, trace, dim_query, dim_scaling, context, res_idx)
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

    # Proceed with expansion of the arguments
    for i, arg in node.node_args.items():
        if is_expandable(arg):
            new_arg = _expand_node(
                arg,
                trace,
                restricted_dims,
                dim_scaling,
                context,
                res_idx,
            )
            new_node.update_arg(i, new_arg)

    context[(node, get_indexed_dims(restricted_dims, node), res_idx)] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    res_idx: int = 0,
) -> CustomOp:
    """Expand a reduction in a specific dimension and recursively proceed to its inputs."""
    # Determine the dimensions to expand the reduction from the indexing of its users
    users = reduction.users
    expand_dims: list[IndexSymbol] = []
    for user in users:
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
        for arg_idx, arg in enumerate(return_vals):
            arg = get_custom(arg)
            # Add GetResult nodes for the corresponding dimensions
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            new_node.add_to_graph(reduction.graph)
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[
                (reduction, get_indexed_dims(dims, expand_dims), arg_idx)
            ] = new_node

            # Proceed with expansion inside the reduction
            new_output_args.append(
                _expand_node(
                    arg,
                    trace,
                    dims,
                    dim_scaling,
                    context,
                    res_idx,
                )
            )

        # Proceed with expansion outside the reduction
        for init_arg in reduction.init_args:
            new_init_args.append(
                _expand_node(
                    get_custom(init_arg),
                    trace,
                    dims,
                    dim_scaling,
                    context,
                    res_idx,
                )
            )

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
        res_idx,
    )
    # Even though we expanded the reduction in multiple dimensions, we only return
    # the node corresponding to the original query
    return context[(reduction, get_indexed_dims(dim_query, expand_dims), res_idx)]


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


def get_dim_scaling(constraints: Sequence[Constraint]) -> dict[IndexSymbol, int]:
    """Get the number of expansions for the dimensions based on the constraints."""
    dim_scaling: dict[IndexSymbol, int] = {}
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
            if not _contains(constraint.dim, hw_cons.vector_shapes):
                raise ValueError(
                    f"Attempting to determine vector shape for unmapped dimension {constraint.dim}"
                )

            vector_size = hw_cons.vector_shapes[constraint.dim]

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


def _handle_reduction_dim(
    reduction: Reduction,
    output: Output,
    trace: CapturedTrace,
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
    res_idx: int,
):
    # Rediscover iter args
    # TODO: Register iter args with the reduction initially so accessing them is easier
    iter_args: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    for node in (get_custom(fx_node) for fx_node in reduction_subgraph.nodes):
        if isinstance(node, IterArg):
            iter_args.append(node)

    new_outputs = list(reduction.outputs(trace.get_subgraph(reduction.subgraph_name)))
    # Users of the loop carried nodes will be duplicated
    for idx, carried_node in enumerate(iter_args):
        # The initial nodes are expanded in the first dimension, so we start from 1
        for scale_idx in range(1, dim_scaling[reduction.axis]):
            for user in carried_node.users:
                if isinstance(user, Output):
                    continue

                dims = dict(user.fx_node.expanded_dims)
                dims[reduction.axis] = scale_idx
                # Temporarily replace the loop carried arg here to avoid
                # duplicated expansion. Otherwise we have the following situation:
                # Suppose we have:
                #   mma_0_0_0(..., acc_0_0_0)
                #   mma_0_0_1(..., mma_0_0_0)
                # Expanding mma_0_0_1 to mma_0_0_2 will trigger expansion of its arg
                # mma_0_0_0 in dims 0_0_2 as well, effectively duplicating the new node.
                # To avoid this we temporarily replace the use of it with a dummy
                # placeholder which will not trigger further expansion.
                index = user.get_node_arg_index(carried_node)
                dummy = Placeholder("dummy").add_to_graph(user.graph)
                dummy.type = None

                saved_arg = user.node_args[index]
                user.update_arg(index, dummy)
                new_node = _expand_node(
                    user,
                    trace,
                    dims,
                    dim_scaling,
                    context,
                    res_idx,
                )

                # This expansion always happens, user should never be reused
                assert new_node != user
                user.update_arg(index, saved_arg)
                new_node.update_arg(index, user)
                user.graph.erase_node(dummy)
                carried_node = user
                new_outputs[idx] = new_node.fx_node

    output.update_arg("return_vals", new_outputs)
