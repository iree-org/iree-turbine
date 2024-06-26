from typing import Any, Optional


import itertools
from ..compiler.ir import Context, Operation
from .constraints import (
    Constraint,
    WorkgroupConstraint,
    get_grid_shape,
)
from .codegen import WaveEmitter
from ..lang import Grid, sym
from ..ops.wave_ops import *
from .._support.tracing import (
    CapturedTrace,
    CompiledContext,
    KernelRegionGraph,
    Launchable,
)
from ..lang.wave_types import Memory, Register, is_memory_meta_derived
import torch.fx as fx

from sympy import Symbol


# Get the last element of the fx node_list structure
def get_last(node_list: fx.graph._node_list) -> fx.Node:
    return next(iter(reversed(node_list)))


def is_expandable(arg: Any) -> bool:
    # Special case for the init args of a reduction node
    if isinstance(arg, Placeholder) and hasattr(arg.fx_node, "reduction_init_arg"):
        return True
    return isinstance(arg, CustomOp) and not isinstance(arg, Placeholder)


dim_scaling = {
    sym.M: 2,
    sym.N: 2,
    sym.K: 2,
}
# sym.BLOCK_M: 2,
# sym.BLOCK_N: 2,
# sym.BLOCK_K: 2,


def get_dim_combinations(all_dims: dict[Symbol, int], selection: Sequence[Symbol]):
    """
    Returns all combinations of sizes for the selected dimensions.
    Other dimensions are clamped to 0.
    """
    adjusted_dimension_sizes = [
        list(range(dim_scaling[dim])) if dim in selection else [0]
        for dim in dim_scaling
    ]
    return itertools.product(*adjusted_dimension_sizes)


def expansion_needed(dims: dict[Symbol, int], selection: Sequence[Symbol]) -> bool:
    return any(dim[1] != 0 and dim[0] in selection for dim in dims.items())


def indexing_dims(all_dims: dict[Symbol, int], nodeOrDims: CustomOp | Sequence[Symbol]):
    if isinstance(nodeOrDims, CustomOp):
        nodeOrDims = nodeOrDims.indexing_dims
    return tuple((key, all_dims[key]) for key in nodeOrDims)


def expand_graph(trace: CapturedTrace, constraints: Optional[list[Constraint]] = None):
    """
    Create a graph that represents the expanded version of the wave function
    """

    # Determine how many nodes there are in the final graph.

    # Start from the back and always expand in the corresponding indexing dimensions
    # Then proceed to the operands

    for node in (
        get_custom(fx_node) for fx_node in reversed(list(trace.get_root_graph().nodes))
    ):
        expansion_context: dict[Any, Any] = {}
        # Start expansion with leaf nodes
        if isinstance(node, Write):
            for dim_vals in get_dim_combinations(dim_scaling, node.indexing_dims):
                dims = {dim: val for dim, val in zip(dim_scaling.keys(), dim_vals)}
                print(f"leaf: {dims}")
                if not expansion_needed(dims, node.indexing_dims):
                    print("skipping cloning")
                    new_node = node
                else:
                    node.graph.inserting_after(node.fx_node)
                    new_node = node.copy()
                    for arg_idx, arg in enumerate(node.node_args):
                        if is_expandable(arg):
                            new_arg = _expand_node(arg, trace, dims, expansion_context)
                            new_node.update_arg(arg_idx, new_arg)
                new_node.fx_node.name = get_suffixed_name(node, dims)
                expansion_context[(node, indexing_dims(dims, node))] = new_node


def get_suffixed_name(node: CustomOp, dims: dict[Symbol, int]) -> str:
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
    context: dict[tuple[CustomOp, dict[Symbol, int]], CustomOp],
) -> CustomOp:
    """Expand a single node in specific dimensions and recursively proceed to its inputs."""

    if (node, indexing_dims(dims, node)) in context:
        print(f"Already expanded node: {node} in {dims}")
        return context[(node, indexing_dims(dims, node))]

    print(f"Expanding node: {node} in {dims}")
    if (
        isinstance(node, Reduction)
        or hasattr(node, "indexing_dims")
        or hasattr(node, "type")
        # and is_memory_meta_derived(node.type)
    ):
        pass
        # Expand in the corresponding dimensions
        if isinstance(node, Reduction):
            return _expand_reduction(node, trace, dims, context)
        else:
            # Do not duplicate the node if it does not index in the expanded dim.
            # We create a duplicate node and reuse the initial node instead of the last copy
            new_node = (
                node.copy() if expansion_needed(dims, node.indexing_dims) else node
            )
            new_node.fx_node.name = get_suffixed_name(node, dims)

            # TODO: adjust indexing when that is modeled

            # TODO: for debugging only
            if new_node == node:
                print(f"did not clone node: {node} in {dims}")
                # new_node.fx_node.name = f"{node.fx_node.name}_{dim[0].name}"
            # TODO: This special case should be handled better
            if isinstance(node, Placeholder) and hasattr(
                node.fx_node, "reduction_init_arg"
            ):
                new_node.fx_node.reduction_init_arg = True

        for i, arg in enumerate(node.node_args):
            if is_expandable(arg):
                new_arg = _expand_node(arg, trace, dims, context)
                new_node.update_arg(i, new_arg)

        context[(node, indexing_dims(dims, node))] = new_node
        return new_node

    print(f"aborted expanding node:{node}, no type info.")
    return node


def _expand_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    initial_dims: dict[Symbol, int],
    context: dict[tuple[CustomOp, Symbol, int], CustomOp],
) -> CustomOp:
    """Expand a reduction in a specific dimension and recursively proceed to its inputs."""
    # TODO: Okay, rework this to expand the reduction completely, not in individual dimensions.

    # Find out in which dimensions to expand this.
    users = reduction.users
    expand_dims: list[Symbol] = []
    for user in users:
        for indexing_dim in user.indexing_dims:
            if indexing_dim not in expand_dims:
                expand_dims.append(indexing_dim)
    print(f"expand dims: {expand_dims}")

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
            result_idx = dim_idx * len(expand_dims) + arg_idx
            reduction.graph.inserting_after(reduction.fx_node)
            new_node = GetResult(reduction.fx_node, result_idx)
            new_node.add_to_graph(reduction.graph)
            if return_node is None:
                return_node = new_node
            new_node.fx_node.name = get_suffixed_name(new_node, dims)
            context[(reduction, indexing_dims(dims, expand_dims))] = new_node

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

    # TODO: Next: What is conceptually missing is the semantics of the reduction.
    # The reduction tells us that for the gemm example the return val of the region
    # is the input to the next iteration of the region. This is not yet captured.
    # Question: Is it always the case that this is done once, i.e. going from 4 MMAs to 8?
    # If not, on which constraints does it depend?
    # So TODO: Build a generic way of doing this, from this function. Possibly
    # simply triggering expansion again from the output.
    # This will only do expansion of the nodes that use the accumulator, as that is the value
    # that in the second go is replaced with the output of the reduction, i.e. the 4 MMAs
    # SO! In the second go this feels like normal expansion where the args of `output`
    # are simply replaced? Not quite sure

    # TODO: New note: the reduction node has a reduction dimension!
    # Does that help us get to the 8 MMAs?

    # TODO: When expanding along the reduction axis, I think we need to remap the accumulator values.
    # I will try first without that and see.

    # TODO: I think the outputs are all the expansions in dimension K?

    return context[(reduction, indexing_dims(initial_dims, expand_dims))]


def _handle_reduction_dim(
    reduction: Reduction,
    output: Placeholder,
    trace: CapturedTrace,
    context: dict[tuple[CustomOp, Symbol, int], CustomOp],
):
    print("Reduction Dim not handled yet.")
    return
    # iter_arg_mapping: dict[CustomOp, CustomOp] = {}
    # Map iter args to output args

    # for idx, arg in enumerate(output.node_args):
    #     iter_arg_mapping[reduction.args[idx]] = arg

    # TODO: Find a better way of getting to the iter args
    iter_args: list[CustomOp] = []
    reduction_subgraph = trace.get_subgraph(reduction.subgraph_name)
    for node in reduction_subgraph.nodes:
        if hasattr(node, "reduction_init_arg"):
            iter_args.append(get_custom(node))

    for idx, iter_arg in enumerate(iter_args):
        for user in iter_arg.users:

            new_node = _expand_node(user, trace, dim_scaling, context)
            # Adjust the arg
            new_node.node_args.index(iter_arg)
            # Brittle: The correct order of the iterargs is not guaranteed

            new_node.update_arg(
                new_node.node_args.index(iter_arg), output.node_args[idx]
            )
            # Updating the output args is not yet handled nicely
            output.fx_node.update_arg(idx, new_node.fx_node)

    # for iter_arg in reduction.iter_args:
    #     for user in iter_arg.users:
    #         pass

    # Users of the iter arg will be duplicated

    # TODO: This will be factored out into its own thing.
    # Expand in the reduction dimension if not already done:
    # if not (reduction, reduction.axis, 0) in context:
    #     for dim_idx in range(dim_scaling[reduction.axis]):
    #         for i, arg in enumerate(output.node_args):
    #             if is_expandable(arg):
    #                 new_arg = _expand_node(
    #                     arg, trace, (reduction.axis, dim_idx), dim_scaling, context
    #                 )
    #                 # Update args of the output node
    #         context[(reduction, reduction.axis, dim_idx)] = reduction


# Currently not needed because we branch for Reduction above
# elif isinstance(arg, list):
#     new_arg = []
#     for sub_arg in arg:
#         # subargs are not CustomOps yet
#         if isinstance(sub_arg, CustomOp):
#             custom_sub_arg = get_custom(sub_arg)
#             new_sub_arg = _expand_node(
#                 custom_sub_arg, trace, dim, dim_scaling, context
#             )
#             new_arg.append(new_sub_arg)
#         else:
#             new_arg.append(sub_arg)
#     new_node.update_arg(i, new_arg)
