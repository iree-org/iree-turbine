import itertools
from sympy import Symbol
import torch.fx as fx
from typing import Any, Optional, TypeAlias

from .constraints import Constraint
from ..lang import sym
from ..ops.wave_ops import *
from .._support.tracing import CapturedTrace
from ...support.logging import get_logger

logger = get_logger("turbine.wave.expansion")
ExpandedNodeMap: TypeAlias = dict[
    tuple[CustomOp, tuple[tuple[Symbol, int], ...]], CustomOp
]
dim_scaling = {
    sym.M: 2,
    sym.N: 2,
    sym.K: 2,
}
# sym.BLOCK_M: 2,
# sym.BLOCK_N: 2,
# sym.BLOCK_K: 2,


def expansion_needed(dims: dict[Symbol, int], selection: Sequence[Symbol]) -> bool:
    """Check if any of the dimensions in the selection are non-zero."""
    return any(dim[1] != 0 and dim[0] in selection for dim in dims.items())


def filter_and_zero_unselected_dims(
    dims: dict[Symbol, int], selection: Sequence[Symbol]
) -> dict[Symbol, int]:
    """
    Filters dimensions based on selection and sets unselected dimensions' values to zero.
    """
    return {dim: val if dim in selection else 0 for dim, val in dims.items()}


def get_dim_combinations(all_dims: dict[Symbol, int], selection: Sequence[Symbol]):
    """
    Returns all combinations of sizes for the selected dimensions.
    Other dimensions are clamped to 0.
    """
    adjusted_dimension_sizes = [
        list(range(all_dims[dim])) if dim in selection else [0] for dim in all_dims
    ]
    return itertools.product(*adjusted_dimension_sizes)


def get_indexed_dims(
    all_dims: dict[Symbol, int], nodeOrDims: CustomOp | Sequence[Symbol]
) -> tuple[tuple[Symbol, int], ...]:
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


def expand_graph(trace: CapturedTrace, constraints: Optional[list[Constraint]] = None):
    """
    Create a graph that represents the expanded version of the wave function.
    The expansion is done in the dimensions specified by the constraints.
    """
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
                            arg, trace, expand_dims, expansion_context
                        )
                        new_node.update_arg(arg_idx, new_arg)
            new_node.fx_node.name = get_expanded_name(node, expand_dims)
            expansion_context[(node, get_indexed_dims(expand_dims, node))] = new_node


def get_expanded_name(node: CustomOp, dims: dict[Symbol, int]) -> str:
    separated = node.fx_node.name.split("_")
    node_name = separated[0]
    if node_name == "get":
        node_name = node_name + separated[1]
    for val in dims.values():
        node_name += f"_{val}"
    return node_name


def _expand_node(
    node: CustomOp,
    trace: CapturedTrace,
    dims: dict[Symbol, int],
    context: ExpandedNodeMap,
) -> CustomOp:
    """Expand a single node in specific dimensions and recursively proceed to its inputs."""
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dims, node)) in context:
        logger.debug(f"Already expanded node: {node} in {dims}")
        return context[(node, get_indexed_dims(dims, node))]
    # Iter args are currently not expanded multiple times.
    elif (
        hasattr(node.fx_node, "expanded_dims")
        and isinstance(node, IterArg)
        and (
            filter_and_zero_unselected_dims(dims, node.indexing_dims)
            == node.fx_node.expanded_dims
        )
    ):
        logger.debug(f"Already expanded node: {node} in {node.fx_node.expanded_dims}")
        return node
    elif isinstance(node, Reduction):
        return _expand_reduction(node, trace, dims, context)

    # Filter out the dimensions that are not indexed by the node
    restricted_dims = filter_and_zero_unselected_dims(dims, node.indexing_dims)
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
            new_arg = _expand_node(arg, trace, restricted_dims, context)
            new_node.update_arg(i, new_arg)

    context[(node, get_indexed_dims(restricted_dims, node))] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    initial_dims: dict[Symbol, int],
    context: ExpandedNodeMap,
) -> CustomOp:
    """Expand a reduction in a specific dimension and recursively proceed to its inputs."""
    # TODO: Think about whether this does the same as the function indexing_dims
    users = reduction.users
    expand_dims: list[Symbol] = []
    for user in users:
        for indexing_dim in user.indexing_dims:
            if indexing_dim not in expand_dims:
                expand_dims.append(indexing_dim)
    logger.debug(f"expanding reduction in dims: {expand_dims}")

    # Get the output node of the reduction
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    output = get_custom(get_last(reduction_subgraph.nodes))

    return_node = None
    new_output_args = []
    new_init_args = []
    for dim_idx, dim_vals in enumerate(get_dim_combinations(dim_scaling, expand_dims)):
        for arg_idx, arg in enumerate(output.node_args):
            dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
            # Add GetResult nodes for the corresponding dimensions
            # result_idx = dim_idx * len(expand_dims) + arg_idx
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            new_node.add_to_graph(reduction.graph)
            if return_node is None:
                return_node = new_node
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[(reduction, get_indexed_dims(dims, expand_dims))] = new_node

            # Proceed with expansion inside the reduction
            new_output_args.append(_expand_node(arg, trace, dims, context))

            # Proceed with expansion outside the reduction
            for init_arg in reduction.init_args:
                new_init_args.append(
                    _expand_node(
                        get_custom(init_arg),
                        trace,
                        dims,
                        context,
                    )
                )

    # Update output node
    output.graph.inserting_before(output.fx_node)
    new_output = output.graph.create_node(
        "output", "output", tuple([node.fx_node for node in new_output_args])
    )
    output.graph.erase_node(output.fx_node)
    # Note: The custom printing of torch.fx does not print the output of multiple args correctly!

    # Update reduction node
    reduction.args = new_init_args
    # TODO: have init_args as property that I can update more easily
    reduction.fx_node.args = (
        reduction.axis,
        [new_init_arg.fx_node for new_init_arg in new_init_args],
        reduction.subgraph_name,
        reduction.implicit_captures,
    )

    _handle_reduction_dim(reduction, get_custom(new_output), trace, context)
    return context[(reduction, get_indexed_dims(initial_dims, expand_dims))]


def _handle_reduction_dim(
    reduction: Reduction,
    output: Placeholder,
    trace: CapturedTrace,
    context: dict[tuple[CustomOp, Symbol, int], CustomOp],
):
    # iter_arg_mapping: dict[CustomOp, CustomOp] = {}
    # Map iter args to output args

    # for idx, arg in enumerate(output.node_args):
    #     iter_arg_mapping[reduction.args[idx]] = arg

    # TODO: Find a better way of getting to the iter args
    iter_args: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    for node in (get_custom(fx_node) for fx_node in reduction_subgraph.nodes):
        if isinstance(node, IterArg):
            iter_args.append(node)

    # Users of the iter arg will be duplicated
    for idx, iter_arg in enumerate(iter_args):
        for user in iter_arg.users:
            dims = {dim: 0 if dim != reduction.axis else 1 for dim in dim_scaling}
            dims = user.fx_node.expanded_dims
            dims[reduction.axis] = 1
            new_node = _expand_node(user, trace, dims, context)
            # Adjust the arg
            new_node.node_args.index(iter_arg)

            # TODO: user is not always an arg to the ouput node.
            # For more complicated kernels we will have to represent this mapping
            # differently.
            outidx = output.node_args.index(user)
            new_node.update_arg(
                new_node.node_args.index(iter_arg), output.node_args[outidx]
            )
            # Updating the output args is not yet handled nicely
            output.fx_node.update_arg(outidx, new_node.fx_node)
