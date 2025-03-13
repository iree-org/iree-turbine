# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...ops.wave_ops import (
    Allocate,
    BinaryPyOp,
    Broadcast,
    CustomOp,
    IterArg,
    MMA,
    NestedRegionOp,
    Output,
    Placeholder,
    Read,
    Reduction,
    Write,
    get_custom,
)
from ..constraints import (
    Constraint,
    HardwareConstraint,
    TilingConstraint,
    WorkgroupConstraint,
)
from ..assumptions import Assumption
from ..symbolic_constraints import SymbolicAlias
from ..._support.tracing import CapturedTrace, IndexingContext
from ..._support.indexing import IndexSymbol, IndexSequence
from ...lang.global_symbols import *
from ..utils import (
    get_mma_dimensional_mapping,
    get_hardware_constraint,
    get_inputs,
    get_users,
    get_largest_index_and_size,
    partial,
    print_trace,
    try_apply_pass,
)
import torch.fx as fx
from typing import Sequence, Callable, Optional
from ....support.logging import get_logger
from copy import deepcopy, copy

logger = get_logger("turbine.wave.index_sequence_analysis")


def combine_derived_index(
    src_index: Optional[dict[IndexSymbol, IndexSequence]],
    dst_index: dict[IndexSymbol, IndexSequence],
) -> dict[IndexSymbol, IndexSequence]:
    if src_index is None:
        return dst_index

    new_index = copy(src_index)
    for dim, new_idx in dst_index.items():
        if dim not in src_index:
            continue

        old_idx = src_index[dim]
        if old_idx == new_idx:
            continue

        assert (
            old_idx.start == 0 or old_idx.start == old_idx.start
        ), f"Index conflict: {old_idx} and {new_idx}"
        new_index[dim] = new_idx

    return new_index


def set_derived_index(trace):
    sources = trace.walk(lambda node: isinstance(get_custom(node), (Read, Write)))

    worklist = []
    for source in sources:
        worklist += get_custom(source).get_derived_indices()

    while len(worklist) > 0:
        current, index = worklist.pop()
        custom = get_custom(current)
        custom.index = combine_derived_index(custom.index, index)
        for inp in get_inputs(current)[0]:
            new_index = custom.transform_index_backwards(custom.index, inp)
            worklist.append((inp, new_index))


def verify_nodes(trace: CapturedTrace, constraints: list[Constraint]):
    """
    Verify that all the valid nodes have their index and vector shapes set.
    """
    nodes = trace.walk(lambda x: x)
    for node in nodes:
        custom = get_custom(node)
        if isinstance(custom, (Placeholder, Allocate)) and not isinstance(
            custom, IterArg
        ):
            continue
        if isinstance(custom, (Output, NestedRegionOp)):
            continue
        assert custom.index, f"Index not set for node {custom.fx_node}"
        if not custom.vector_shapes:
            # If vector_shapes is not set, see if it can be derived from the hardware constraints.
            hw_constraint = get_hardware_constraint(constraints)
            update_vector_shapes = [
                dim for dim in custom.index if dim in hw_constraint.vector_shapes
            ]
            if update_vector_shapes:
                custom.vector_shapes = {}
                for dim in update_vector_shapes:
                    custom.vector_shapes[dim] = hw_constraint.vector_shapes[dim]
        assert custom.vector_shapes, f"Vector shapes not set for node {custom.fx_node}"


def set_node_indices(
    trace: CapturedTrace,
    constraints: list[Constraint],
    print_ir_before: Sequence[str] = [],
    print_ir_after: Sequence[str] = [],
):
    mma_mapping = get_mma_dimensional_mapping(
        trace, get_hardware_constraint(constraints)
    )
    trace.walk(partial(set_thread_independent_index, constraints))

    if (
        "all" in print_ir_after
        or "all" in print_ir_before
        or "trace" in print_ir_after
        or "first" in print_ir_before
    ):
        print(
            f"***After set_thread_independent_index/Before set_thread_dependent_index pass***\n"
        )
        print_trace(trace)

    graph_passes = []
    if mma_mapping != {}:
        graph_passes += [
            partial(
                set_thread_dependent_index_from_mma, constraints, mma_mapping, trace
            )
        ]
    else:
        graph_passes += [
            partial(set_thread_dependent_index_from_read_write, constraints, trace)
        ]
    graph_passes += [
        partial(set_derived_index, trace),
        partial(resolve_thread_shapes, trace, constraints),
        partial(verify_nodes, trace, constraints),
    ]
    for p in graph_passes:
        try_apply_pass(p, trace, print_ir_before, print_ir_after)


def compute_stride(
    symbolic_shape: tuple[IndexSymbol, ...],
    vector_shapes: dict[IndexSymbol, int],
    target_dim: IndexSymbol,
) -> int:
    """
    Compute the stride for a given dimension based on the vector shapes.
    The stride is the product of the vector shapes of all dimensions that are
    not the given dimension.
    """
    stride = 1
    for dim in reversed(symbolic_shape):
        if dim == target_dim:
            break
        assert dim in vector_shapes, f"Dimension {dim} not found in vector shapes"
        stride *= vector_shapes[dim]

    try:
        stride = int(stride)
    except Exception as e:
        logger.error(e)
    return stride


def is_contiguous_dim(
    dim: IndexSymbol, symbolic_shape: list[IndexSymbol], vector_shapes: list[int]
) -> bool:
    """
    Checks if the given dimension is stored contiguously in memory. This happens if
    the dimension is the last one in the symbolic shape or all dimensions after it
    are unit dimensions.
    """
    is_innermost_dim = dim == symbolic_shape[-1]
    dim_index = symbolic_shape.index(dim)
    static_shape = [vector_shapes[dim] for dim in symbolic_shape]
    all_unit_dims = all(dim == 1 for dim in static_shape[dim_index + 1 :])
    return is_innermost_dim or all_unit_dims


def set_thread_independent_index(
    constraints: Sequence[Constraint],
    node: fx.Node,
):
    """
    Set the index of the node based on all constraints except the hardware constraint.
    """
    custom = get_custom(node)
    if isinstance(custom, (Reduction, Placeholder)) and not isinstance(custom, IterArg):
        return

    hw_cons = get_hardware_constraint(constraints)
    constraints = [
        c
        for c in constraints
        if not isinstance(c, (HardwareConstraint, Assumption, SymbolicAlias))
    ]

    index = {}
    for dim in custom.indexing_dims:
        index_seq = None
        for constraint in constraints:
            if constraint.dim != dim:
                continue

            # If the constraint is a tiling constraint, and the node
            # is outside a reduction, we don't apply the constraint.
            if isinstance(constraint, TilingConstraint):
                if not hasattr(custom.graph, "parent_op"):
                    continue

            if index_seq is None:
                index_seq = constraint.apply()
            else:
                index_seq.start += constraint.apply().start

        if index_seq is not None:
            index.update({dim: index_seq})
        else:
            index.update({dim: IndexSequence(0, 1, 1)})

    custom.index = index


def specialize_index(
    index: dict[IndexSymbol, IndexSequence], subs: dict[IndexSymbol, int]
):
    """
    Specialize the index sequence with the given substitutions.
    """
    return {dim: seq.subs(subs) for dim, seq in index.items()}


def populate_mma_source_indices(
    node: MMA,
    mma_index: dict[MMA, dict[IndexSymbol, int]],
    hardware_constraint: HardwareConstraint,
):
    """
    Initialize the sources with the LHS, RHS, ACC and MMA node
    and their index sequences and vector shapes. These will
    be propagated to the rest of the graph.
    """
    index: dict[IndexSymbol, IndexSequence] = {}
    mapping = mma_index[node]
    for dim, dim_index in mapping.items():
        index[dim] = hardware_constraint.apply_mma_mapping(
            dim, dim_index, node.mma_type
        )
    node.index = combine_indices(node.index, index)
    return [
        (
            get_custom(node.lhs),
            specialize_index(index, {MMA_LHS: 1, MMA_RHS: 0, MMA_ACC: 0}),
            node.vector_shapes,
        ),
        (
            get_custom(node.rhs),
            specialize_index(index, {MMA_LHS: 0, MMA_RHS: 1, MMA_ACC: 0}),
            node.vector_shapes,
        ),
        (
            get_custom(node.acc),
            specialize_index(index, {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}),
            node.vector_shapes,
        ),
        (
            node,
            specialize_index(index, {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}),
            node.vector_shapes,
        ),
    ]


def populate_read_write_source_indices(
    node: Read | Write,
    hardware_constraint: HardwareConstraint,
    workgroup_constraints: list[WorkgroupConstraint],
):
    """
    Initialize the sources with the read and/or write nodes
    and their index sequences and vector shapes. These will
    be propagated to the rest of the graph.
    """
    index: dict[IndexSymbol, IndexSequence] = {}
    for dim in node.indexing_dims:
        elements_per_thread = (
            1
            if not is_contiguous_dim(
                dim, node.indexing_dims, hardware_constraint.vector_shapes
            )
            else node.elements_per_thread
        )
        stride = compute_stride(
            node.indexing_dims, hardware_constraint.vector_shapes, dim
        )
        wg_constraint = [x for x in workgroup_constraints if x.dim == dim]
        if not wg_constraint:
            continue
        index[dim] = hardware_constraint.apply_read_write_thread_mapping(
            dim, wg_constraint[0].workgroup_dim, elements_per_thread, stride
        )
    return [(node, index, hardware_constraint.vector_shapes)]


def combine_indices(
    thread_independent_index: dict[IndexSymbol, IndexSequence],
    thread_dependent_index: dict[IndexSymbol, IndexSequence],
) -> dict[IndexSymbol, IndexSequence]:
    """
    The thread dependent index is obtained from "anchor" nodes like MMA, Read, Write
    which make the index sequence (access pattern) thread specific. These are
    added to the thread independent index which is obtained from the constraints.
    """
    combined_index = {k: v for k, v in thread_independent_index.items()}
    for k in combined_index:
        if k in thread_dependent_index:
            combined_index[k].start += thread_dependent_index[k].start
            combined_index[k].size = thread_dependent_index[k].size
            combined_index[k].stride = thread_dependent_index[k].stride
    return combined_index


def add_nodes_to_sources(
    source: CustomOp,
    reduction: Reduction,
    fn: Callable,
    source_index: dict[IndexSymbol, IndexSequence],
    source_vector_shapes: dict[IndexSymbol, int],
    sources: list[
        tuple[CustomOp, dict[IndexSymbol, IndexSequence], dict[IndexSymbol, int]]
    ],
) -> tuple[list[CustomOp], Reduction]:
    """
    Populate the sources with the inputs and users of the source node.
    """
    for args, reduction in [fn(source.fx_node, reduction)]:
        logger.debug(f"{source.fx_node} -> {args}")
        if not args:
            break
        for arg in args:
            custom = get_custom(arg)
            if isinstance(custom, (Allocate, Placeholder)) and not isinstance(
                custom, IterArg
            ):
                continue
            vector_shapes = (
                custom.vector_shapes if custom.vector_shapes else source_vector_shapes
            )
            sources.append((custom, source_index, vector_shapes))
    return sources, reduction


def should_update_index(
    source: CustomOp,
    source_index: dict[IndexSymbol, IndexSequence],
    source_vector_shapes: dict[IndexSymbol, int],
    symbolic_constraints: list[SymbolicAlias],
):
    # Get symbolic shape without any aliased variables.
    aliased_dims = [x.source for x in symbolic_constraints]
    symbolic_shape = set(source.type.symbolic_shape).difference(aliased_dims)

    # If all the source indexing dimensions are not present in source vector shapes,
    # we should not update the index.
    if not set(symbolic_shape).issubset(set(source_vector_shapes.keys())):
        return False

    # Determine if we should update the idx based on the source.
    # We update the source only if the source index provides
    # information about all the non-batch dimensions of the source.
    non_batch_dims = [x for x in symbolic_shape if source_vector_shapes[x] > 1]

    # If the source index is smaller than the non-batch dims, check if the
    # source index is a subset of the non-batch dims.
    if len(source_index.keys()) < len(non_batch_dims):
        return set(source_index.keys()).issubset(set(non_batch_dims))

    # Otherwise, check if the non-batch dims are a subset of the source index.
    if not set(non_batch_dims).issubset(set(source_index.keys())):
        return False

    return True


def append_aliased_shapes(source: CustomOp, symbolic_constraints: list[SymbolicAlias]):
    """
    Append the aliased shapes to the vector shapes of the source, if they
    are present in the source index.
    """
    for constraint in symbolic_constraints:
        if (
            constraint.target in source.vector_shapes
            and constraint.source in source.index
        ):
            source.vector_shapes[constraint.source] = constraint.apply(
                source.vector_shapes[constraint.target]
            )


def propagate_indices(
    sources: set[CustomOp],
    visited: set[CustomOp],
    symbolic_constraints: list[SymbolicAlias],
):
    """
    Propagate the index and vector shapes through the graph
    starting with priveleged nodes (like MMA, Read, Write).
    """
    reduction = None
    while sources:
        source, source_index, source_vector_shapes = sources.pop(0)
        if source in visited:
            continue
        if not isinstance(source, (NestedRegionOp, MMA)):
            if not should_update_index(
                source, source_index, source_vector_shapes, symbolic_constraints
            ):
                continue
            source_index = source.transform_index(source_index)
            source.index = combine_indices(source.index, source_index)
            source.vector_shapes = source_vector_shapes
            append_aliased_shapes(source, symbolic_constraints)
        visited.add(source)
        for func in [get_inputs, get_users]:
            sources, reduction = add_nodes_to_sources(
                source,
                reduction,
                func,
                source_index,
                source_vector_shapes,
                sources,
            )
    return visited


def set_thread_dependent_index_from_mma(
    constraints: Sequence[Constraint],
    mma_mapping: dict[MMA, dict[IndexSymbol, int]],
    trace: CapturedTrace,
):
    """
    Set the thread dependent index based on the hardware constraint.
    """
    hardware_constraint = get_hardware_constraint(constraints)
    sources: list[MMA] = list(mma_mapping.keys())
    assert sources and len(sources) >= 1, "Unexpected empty MMA mapping."
    if not sources:
        sources = trace.walk(lambda node: isinstance(get_custom(node), (Read, Write)))
        sources = [get_custom(x) for x in sources]
        assert sources, "No read or mma nodes found in the graph."

    visited = set()
    symbolic_constraints = [c for c in constraints if isinstance(c, SymbolicAlias)]
    for source in sources:
        visited = visited.union(set([x for x in sources]))
        visited.remove(source)
        new_sources = populate_mma_source_indices(
            source, mma_mapping, hardware_constraint
        )
        visited = propagate_indices(
            new_sources,
            visited,
            symbolic_constraints,
        )


def set_thread_dependent_index_from_read_write(
    constraints: Sequence[Constraint],
    trace: CapturedTrace,
):
    """
    Set the thread dependent index based on the hardware constraint.
    """
    hardware_constraint = get_hardware_constraint(constraints)
    sources = trace.walk(lambda node: isinstance(get_custom(node), (Read, Write)))
    sources = [get_custom(x) for x in sources]
    assert sources, "No read nodes found in the graph."

    visited = set()
    workgroup_constraints = [
        c for c in constraints if isinstance(c, WorkgroupConstraint)
    ]
    symbolic_constraints = [c for c in constraints if isinstance(c, SymbolicAlias)]
    for source in sources:
        visited = visited.union(set([x for x in sources]))
        visited.remove(source)
        new_sources = populate_read_write_source_indices(
            source, hardware_constraint, workgroup_constraints
        )
        visited = propagate_indices(
            new_sources,
            visited,
            symbolic_constraints,
        )


def set_post_expansion_indices(trace: CapturedTrace, constraints: list[Constraint]):
    """
    Add offsets to the indices based on the expanded dims.
    """

    def apply_offset(node: fx.Node):
        custom = get_custom(node)
        if custom.expanded_dims is None:
            return False
        for dim, scale in custom.expanded_dims.items():
            if dim in custom.index:
                custom.index[dim].start += scale * custom.vector_shapes[dim]
        return False

    trace.walk(apply_offset)


def create_broadcast(
    binary_op: BinaryPyOp,
    to_broadcast: CustomOp,
    broadcast_dim: IndexSymbol,
    broadcast_size: int,
    target_node: CustomOp,
):
    """
    Create a broadcast node for the given binary operator.
    """
    with binary_op.graph.inserting_before(binary_op.fx_node):
        broadcasted = Broadcast(
            to_broadcast.fx_node, target_node.type.symbolic_shape
        ).add_to_graph(binary_op.graph)
        custom = get_custom(broadcasted)
        custom.vector_shapes = binary_op.vector_shapes
        custom.index = deepcopy(target_node.index)
        custom.index[broadcast_dim].size = broadcast_size
        broadcast_idx = list(binary_op.node_args.values()).index(to_broadcast)
        binary_op.update_arg(broadcast_idx, custom.fx_node)


def resolve_thread_shapes(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function walks through all the binary operators in the graph and
    if there is a discrepancy between the thread shapes of the operators
    along the same dimension it resolves the discrepancy.

    Currently, the only mismatches that can be resolved are when one of
    the shapes is 1 and the other is > 1.
    """

    def get_index(custom: CustomOp):
        if isinstance(custom, MMA):
            return custom.acc.index
        return custom.index

    binary_ops = trace.walk(lambda node: isinstance(get_custom(node), BinaryPyOp))
    for binary_op in binary_ops:
        custom = get_custom(binary_op)
        # Get the largest dim and shape from the lhs and rhs.
        lhs = get_custom(custom.lhs)
        rhs = get_custom(custom.rhs)

        lhs_dim, lhs_size = get_largest_index_and_size(get_index(lhs))
        rhs_dim, rhs_size = get_largest_index_and_size(get_index(rhs))

        # If they are equal we are done.
        if lhs_dim == rhs_dim and lhs_size == rhs_size:
            continue
        # If all are unit dims, there is nothing to do.
        if lhs_size == 1 and rhs_size == 1:
            continue
        # Cannot handle discrepancies when both shapes are > 1.
        if lhs_size > 1 and rhs_size > 1:
            raise NotImplementedError(
                "Currently only support resolving discrepancies when one of the shapes is 1."
            )

        broadcast_rhs = lhs_size > rhs_size
        to_broadcast = rhs if broadcast_rhs else lhs
        broadcast_dim = lhs_dim if broadcast_rhs else rhs_dim
        broadcast_size = lhs_size if broadcast_rhs else rhs_size
        broadcasted = lhs if broadcast_rhs else rhs

        if lhs_dim != rhs_dim:
            # If the dimensions don't agree, we can still do this broadcast only if
            # the two nodes differ in shape along the broadcasting dimension and the
            # broadcasting dimension is the innermost dimension.
            missing_dims = set(broadcasted.type.symbolic_shape).difference(
                set(to_broadcast.type.symbolic_shape)
            )
            is_only_missing_dim = missing_dims == {broadcast_dim}
            is_innermost_dim = broadcast_dim == broadcasted.type.symbolic_shape[-1]

            if not is_only_missing_dim and not is_innermost_dim:
                raise NotImplementedError(
                    "Currently only support resolving discrepancies when the broadcasting dimension is the innermost dimension."
                )

        # Broadcast
        create_broadcast(
            custom, to_broadcast, broadcast_dim, broadcast_size, broadcasted
        )
