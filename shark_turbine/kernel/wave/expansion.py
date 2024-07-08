import itertools
from sympy import Symbol
import torch.fx as fx
from typing import Any, TypeAlias

from .constraints import Constraint, HardwareConstraint, WorkgroupConstraint
from ..ops.wave_ops import *
from .._support.indexing import IndexingContext
from ...support.logging import get_logger
from .._support.tracing import CapturedTrace

logger = get_logger("turbine.wave.expansion")
ExpandedNodeMap: TypeAlias = dict[
    tuple[CustomOp, tuple[tuple[IndexSymbol, int], ...]], CustomOp
]


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
    all_dims: dict[IndexSymbol, int], selection: Sequence[IndexSymbol]
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
    return tuple((key, all_dims[key]) for key in nodeOrDims)


def get_last(node_list: fx.graph._node_list) -> fx.Node:  # type: ignore
    """Get the last element of the fx node_list structure"""
    return next(iter(reversed(node_list)))  # type: ignore


def is_expandable(arg: Any) -> bool:
    """Check if an argument is expandable."""
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

    expansion_context: ExpandedNodeMap = {}
    for node in (
        get_custom(fx_node) for fx_node in reversed(list(trace.get_root_graph().nodes))
    ):
        # Expansion begins at the leaf nodes
        if node.__class__ not in leaf_nodes:
            continue

        for dim_combination in get_dim_combinations(dim_scaling, node.indexing_dims):
            expand_dims = {
                dim: val for dim, val in zip(dim_scaling.keys(), dim_combination)
            }
            logger.debug(f"Starting expansion at leaf:{node} in dims:{expand_dims}")
            if not expansion_needed(expand_dims, node.indexing_dims):
                new_node = node
            else:
                node.graph.inserting_after(node.fx_node)
                new_node = node.copy()
                for arg_idx, arg in enumerate(node.node_args):
                    if is_expandable(arg):
                        new_arg = _expand_node(
                            arg, trace, expand_dims, dim_scaling, expansion_context
                        )
                        new_node.update_arg(arg_idx, new_arg)
            new_node.fx_node.name = get_expanded_name(node, expand_dims)
            expansion_context[(node, get_indexed_dims(expand_dims, node))] = new_node


def _expand_node(
    node: CustomOp,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
) -> CustomOp:
    """Expand a single node in specific dimensions and recursively proceed to its inputs."""
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dim_query, node)) in context:
        logger.debug(f"Already expanded node: {node} in {dim_query}")
        return context[(node, get_indexed_dims(dim_query, node))]
    # Iter args are not expanded multiple times.
    elif (
        hasattr(node.fx_node, "expanded_dims")
        and isinstance(node, IterArg)
        and (
            filter_and_zero_unselected_dims(dim_query, node.indexing_dims)
            == node.fx_node.expanded_dims
        )
    ):
        logger.debug(f"Already expanded node: {node} in {node.fx_node.expanded_dims}")
        return node
    elif isinstance(node, Reduction):
        return _expand_reduction(node, trace, dim_query, dim_scaling, context)

    # Filter out the dimensions that are not indexed by the node
    restricted_dims = filter_and_zero_unselected_dims(dim_query, node.indexing_dims)
    logger.debug(f"Expanding node: {node} in {restricted_dims}")

    # Clone the node for the new expansion. The original node is reused for the
    # case of all dimensions being zero.
    if expansion_needed(restricted_dims, node.indexing_dims):
        new_node = node.copy()
    else:
        new_node = node
        logger.debug(f"did not clone node: {node} in {restricted_dims}")

    new_node.fx_node.expanded_dims = restricted_dims
    new_node.fx_node.name = get_expanded_name(node, restricted_dims)

    # Proceed with expansion of the arguments
    for i, arg in enumerate(node.node_args):
        if is_expandable(arg):
            new_arg = _expand_node(arg, trace, restricted_dims, dim_scaling, context)
            new_node.update_arg(i, new_arg)

    context[(node, get_indexed_dims(restricted_dims, node))] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    context: ExpandedNodeMap,
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
    for dim_idx, dim_vals in enumerate(get_dim_combinations(dim_scaling, expand_dims)):
        for arg_idx, arg in enumerate(output.node_args):
            dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
            # Add GetResult nodes for the corresponding dimensions
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            new_node.add_to_graph(reduction.graph)
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[(reduction, get_indexed_dims(dims, expand_dims))] = new_node

            # Proceed with expansion inside the reduction
            new_output_args.append(_expand_node(arg, trace, dims, dim_scaling, context))

            # Proceed with expansion outside the reduction
            for init_arg in reduction.init_args:
                new_init_args.append(
                    _expand_node(
                        get_custom(init_arg),
                        trace,
                        dims,
                        dim_scaling,
                        context,
                    )
                )

    # Update init_args and return values
    reduction.update_arg(
        "init_args", [new_init_arg.fx_node for new_init_arg in new_init_args]
    )
    output.update_arg("return_vals", [node.fx_node for node in new_output_args])
    _handle_reduction_dim(reduction, output, trace, dim_scaling, context)
    # Even though we expanded the reduction in multiple dimensions, we only return
    # the node corresponding to the original query
    return context[(reduction, get_indexed_dims(dim_query, expand_dims))]


def get_expanded_name(node: CustomOp, dims: dict[IndexSymbol, int]) -> str:
    """Returns the name of a node with the dimensions appended."""

    separated = node.fx_node.name.split("_")
    node_name = separated[0]
    # Special case for get_result op
    if node_name == "get":
        node_name = node_name + separated[1]
    for val in dims.values():
        node_name += f"_{val}"
    return node_name


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
        if isinstance(constraint, WorkgroupConstraint):
            tile_size = idxc.get_static_value(constraint.tile_size)
            wave_count = hardware_constraints[0].waves_per_block[
                constraint.workgroup_dim
            ]
            mma_size = hardware_constraints[0].mma_matrix_shapes[
                constraint.workgroup_dim
            ]
            if tile_size is None or wave_count is None or mma_size is None:
                raise ValueError(
                    "Tile size, wave count and mma size must be statically known"
                )
            dim_scaling[constraint.dim] = tile_size // wave_count // mma_size
    return dim_scaling


def _handle_reduction_dim(
    reduction: Reduction,
    output: Output,
    trace: CapturedTrace,
    dim_scaling: dict[IndexSymbol, int],
    context: dict[tuple[CustomOp, Symbol, int], CustomOp],
):
    # Rediscover iter args
    # TODO: Register iter args with the reduction initially so accessing them is easier
    iter_args: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    for node in (get_custom(fx_node) for fx_node in reduction_subgraph.nodes):
        if isinstance(node, IterArg):
            iter_args.append(node)

    new_outputs = [iter_arg.fx_node for iter_arg in iter_args]
    # Users of the loop carried nodes will be duplicated
    for idx, carried_node in enumerate(iter_args):
        # The initial nodes are expanded in the first dimension, so we start from 1
        for scale_idx in range(1, dim_scaling[reduction.axis]):
            for user in carried_node.users:
                if isinstance(user, Output):
                    continue

                dims = user.fx_node.expanded_dims
                dims[reduction.axis] = scale_idx
                # Temporarily replace the carried arg here to avoid duplicated expansion.
                # Otherwise we have the following situation:
                # Suppose we have:
                #   mma_0_0_0(..., acc_0_0_0)
                #   mma_0_0_1(..., mma_0_0_0)
                # Expanding mma_0_0_1 to mma_0_0_2 will trigger expansion of its arg
                # mma_0_0_0 in dims 0_0_2 as well, effectively duplicating the new node.
                # To avoid this we temporarily replace the use of it with the original iter_arg
                # Another option would be to pass a mask of which args to expand, but that
                # complicates the expansion base case
                saved_arg = user.node_args[2]
                user.update_arg(2, iter_args[idx])
                new_node = _expand_node(user, trace, dims, dim_scaling, context)
                user.update_arg(2, saved_arg)

                # This expansion always happens, user should never be reused
                assert new_node != user
                new_node.update_arg(
                    2,
                    user,  # output.return_vals[outidx]
                )
                carried_node = user
                new_outputs[idx] = new_node.fx_node

    output.update_arg("return_vals", new_outputs)
