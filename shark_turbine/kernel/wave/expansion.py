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
from .._support.indexing import index_symbol
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.expansion")
# This represents a mapping of a node + indexing into the dimensions to the
# corresponding expanded node in these specific dimensions. An example for a
# record in this map is (read_0_0_0, ((M,0),(N,0),(K,1)) -> read_0_0_1
ExpandedNodeMap: TypeAlias = dict[
    tuple[CustomOp, tuple[tuple[IndexSymbol, int], ...]], CustomOp
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


def get_mma_dimensional_mapping(trace: CapturedTrace) -> dict[IndexSymbol, int]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2).
    """

    def is_mma(node: fx.Node) -> bool:
        custom = get_custom(node)
        return isinstance(custom, MMA)

    mma_nodes = trace.walk(is_mma)
    mapping: dict[IndexSymbol, int] = {}
    for node in mma_nodes:
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        mapping[m] = 0
        mapping[n] = 1
        k = [
            dim
            for dim in custom.lhs_type.symbolic_shape
            if dim in custom.rhs_type.symbolic_shape
            and not dim in custom.acc_type.symbolic_shape
        ][0]
        mapping[k] = 2

    return mapping


def set_node_index(
    constraints: Sequence[Constraint],
    mma_index: dict[IndexSymbol, int],
    dim_tile_size: dict[IndexSymbol, int],
    custom: CustomOp,
    dim_scaling: dict[IndexSymbol, int],
):
    """
    Set the index of the node based on the user constraints. In certain
    operators (like read, write), there is only a single index associated
    with the node (the index to read from, the index to write to). But for
    other operators like mma, each operand reads from a different index.

    Rather than maintain operand specific indices for operators, we maintain
    dimension specific indices for each operator. So for an mma operator that
    has a signature of (MxK, NxK) -> MxN, we maintain only 3 mappings for
    dimensions M, N and K, but allow each mapping to be piecewise conditioned
    on the operand.
    """
    hardware_constraint = [c for c in constraints if isinstance(c, HardwareConstraint)]
    other_constraints = [
        c for c in constraints if not isinstance(c, HardwareConstraint)
    ]
    # Apply hardware constraint first since it dictates the stride and size.
    sorted_constraints = hardware_constraint + other_constraints

    for dim in custom.indexing_dims:
        index_seq = None
        for constraint in sorted_constraints:
            mma_check = (
                isinstance(constraint, HardwareConstraint)
                and dim in mma_index
                and isinstance(custom, MMA)
            )

            constraint_check = (
                not isinstance(constraint, HardwareConstraint) and dim == constraint.dim
            )

            if (not mma_check) and (not constraint_check):
                continue

            if isinstance(constraint, HardwareConstraint):
                index_seq = constraint.apply(mma_index[dim])
            else:
                if index_seq is None:
                    index_seq = constraint.apply()
                else:
                    index_seq.start += constraint.apply().start

        if index_seq is not None:
            if dim in dim_scaling and dim in dim_tile_size:
                index_seq.start += dim_scaling[dim] * dim_tile_size[dim]
            custom.index = {dim: index_seq}
        else:
            custom.index = {dim: IndexSequence(0, 1, 1)}

    setattr(custom.fx_node, "index", custom.index)


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
        node_index_setter = lambda *args: None
    else:
        mma_index = get_mma_dimensional_mapping(trace)
        dim_scaling, dim_tile_size = get_dim_scaling(constraints_or_scaling, mma_index)
        node_index_setter = partial(
            set_node_index, constraints_or_scaling, mma_index, dim_tile_size
        )

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
                node_index_setter,
                expansion_context,
            )


def _expand_node(
    node: CustomOp | list[CustomOp],
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
    context: ExpandedNodeMap,
) -> CustomOp:
    """Expand a single node or list of nodes in specific dimensions and recursively proceed to its inputs."""
    if isinstance(node, list):
        expanded_nodes = []
        for elem in node:
            expanded_nodes.append(
                _expand_node(
                    elem, trace, dim_query, dim_scaling, node_index_setter, context
                ).fx_node
            )
        return expanded_nodes
    # If we expanded a node in the same dimensions before, we can reuse it
    if (node, get_indexed_dims(dim_query, node)) in context:
        logger.debug(f"Already expanded node: {node} in {dim_query}")
        return context[(node, get_indexed_dims(dim_query, node))]
    elif isinstance(node, Reduction):
        return _expand_reduction(
            node, trace, dim_query, dim_scaling, node_index_setter, context
        )
    elif isinstance(node, GetResult):
        # The presence of a GetResult node indicates that the reduction has already
        # been expanded. Simply return the corresponding node.
        reduction = get_custom(node.value)
        return context[(reduction, get_indexed_dims(dim_query, reduction))]
    elif isinstance(node, Allocate):
        # Allocate nodes are not expanded.
        return node

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
    node_index_setter(new_node, restricted_dims)

    # Proceed with expansion of the arguments
    for i, arg in node.node_args.items():
        if is_expandable(arg):
            new_arg = _expand_node(
                arg,
                trace,
                restricted_dims,
                dim_scaling,
                node_index_setter,
                context,
            )
            new_node.update_arg(i, new_arg)

    context[(node, get_indexed_dims(restricted_dims, node))] = new_node
    return new_node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    dim_query: dict[IndexSymbol, int],
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
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
    for dim_vals in get_dim_combinations(dim_scaling, expand_dims):
        for arg_idx, arg in output.node_args.items():
            dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
            # Add GetResult nodes for the corresponding dimensions
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, len(new_output_args))
            new_node.add_to_graph(reduction.graph)
            new_node.fx_node.name = get_expanded_name(new_node, dims)
            context[(reduction, get_indexed_dims(dims, expand_dims))] = new_node

            # Proceed with expansion inside the reduction
            new_output_args.append(
                _expand_node(arg, trace, dims, dim_scaling, node_index_setter, context)
            )

            # Proceed with expansion outside the reduction
            for init_arg in reduction.init_args:
                new_init_args.append(
                    _expand_node(
                        get_custom(init_arg),
                        trace,
                        dims,
                        dim_scaling,
                        node_index_setter,
                        context,
                    )
                )

    # Update init_args and return values
    reduction.update_arg(
        "init_args", [new_init_arg.fx_node for new_init_arg in new_init_args]
    )
    output.update_arg("return_vals", [node.fx_node for node in new_output_args])
    _handle_reduction_dim(
        reduction, output, trace, dim_scaling, node_index_setter, context
    )
    # Even though we expanded the reduction in multiple dimensions, we only return
    # the node corresponding to the original query
    return context[(reduction, get_indexed_dims(dim_query, expand_dims))]


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


def get_dim_scaling(
    constraints: Sequence[Constraint], mma_indices: dict[IndexSymbol, int]
) -> tuple[dict[IndexSymbol, int]]:
    """Get the number of expansions for the dimensions based on the constraints."""
    dim_scaling: dict[IndexSymbol, int] = {}
    dim_tile_size: dict[IndexSymbol, int] = {}
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
            if not (
                constraint.dim in mma_indices or constraint.dim in hw_cons.vector_shapes
            ):
                raise ValueError(
                    f"Attempting to determine vector shape for unmapped dimension {constraint.dim}"
                )

            if mma_indices:
                vector_size = hardware_constraints[0].mma_matrix_shapes[
                    mma_indices[constraint.dim]
                ]
            else:
                vector_size = hardware_constraints[0].vector_shapes[constraint.dim]

            wave_count = 1
            if isinstance(constraint, WorkgroupConstraint):
                wave_count = hardware_constraints[0].waves_per_block[
                    constraint.workgroup_dim
                ]
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
            dim_tile_size[constraint.dim] = vector_size
    return (dim_scaling, dim_tile_size)


def _handle_reduction_dim(
    reduction: Reduction,
    output: Output,
    trace: CapturedTrace,
    dim_scaling: dict[IndexSymbol, int],
    node_index_setter: Callable[[CustomOp, dict[IndexSymbol, int]], None],
    context: ExpandedNodeMap,
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

                dims = user.fx_node.expanded_dims
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
                    user, trace, dims, dim_scaling, node_index_setter, context
                )

                # This expansion always happens, user should never be reused
                assert new_node != user
                user.update_arg(index, saved_arg)
                new_node.update_arg(index, user)
                user.graph.erase_node(dummy)
                carried_node = user
                new_outputs[idx] = new_node.fx_node

    output.update_arg("return_vals", new_outputs)
