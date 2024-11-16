# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ops.wave_ops import (
    Allocate,
    Read,
    Write,
    ExtractSlice,
    get_custom,
    Reduction,
    MMA,
    Placeholder,
    IterArg,
    CustomOp,
    Reshape,
)
from .constraints import Constraint, HardwareConstraint, WorkgroupConstraint
from .assumptions import Assumption
from .._support.tracing import CapturedTrace, IndexingContext
from .._support.indexing import IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from .utils import (
    simplify_index,
    get_mma_dimensional_mapping,
    get_hardware_constraint,
    subs_idxc,
    specialize_index_sequence,
    capture_backward_slice,
)
import torch.fx as fx
import numpy as np
from functools import partial
from typing import Sequence
from ...support.logging import get_logger
import sympy

logger = get_logger("turbine.wave.index_sequence_analysis")


def get_vector_shape(
    vector_shapes: dict[IndexSymbol, int],
    symbolic_shape: list[IndexSymbol],
) -> list[int]:
    vector_shapes = [max(vector_shapes[dim], 1) for dim in symbolic_shape]
    return vector_shapes


def partition_strided_operators(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of operators in the graph
    that are writes on 2d tensors. If the operator has an access pattern where
    the strides are greater than one on a single dimension, this function splits the
    operands into individual elements and constructs a write for
    each individual element.
    """

    def has_strided_access(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if isinstance(custom, Write):
            strides = [simplify_index(custom.index[dim]).stride for dim in custom.index]
            elements_per_thread = [
                simplify_index(custom.index[dim]).size for dim in custom.index
            ]
            strides = [x for x, y in zip(strides, elements_per_thread) if y > 1]
            num_strided_accesses = sum(1 for stride in strides if stride > 1)
            if num_strided_accesses > 1:
                raise NotImplementedError(
                    "Support for strided accesses on more than one dimension not implemented yet!"
                )
            return num_strided_accesses == 1
        return False

    strided_operators = trace.walk(has_strided_access)
    hw_constraint = [c for c in constraints if isinstance(c, HardwareConstraint)][0]
    for operator in strided_operators:
        custom = get_custom(operator)
        simplified_index = {
            dim: simplify_index(custom.index.get(dim, custom.index[dim]))
            for dim in custom.index
        }

        shape = get_vector_shape(
            custom.vector_shapes, custom.register_type.symbolic_shape
        )
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        max_stride_dim, max_stride = max(
            [(dim, seq.stride) for dim, seq in simplified_index.items()],
            key=lambda item: item[1],
        )
        ops_to_combine = []
        with custom.graph.inserting_before(operator):
            for i in range(elements_per_thread):
                # Non-contiguous access patterns can have varying offsets. We
                # handle that here.
                extract = ExtractSlice(custom.register_, [i], [1], [1]).add_to_graph(
                    custom.graph
                )
                offset = np.unravel_index(int(i * max_stride), shape)
                write = Write(
                    extract,
                    custom.memory,
                    mapping=custom.mapping,
                    elements_per_thread=1,
                ).add_to_graph(custom.graph)
                write.index = {
                    dim: IndexSequence(
                        simplified_index[dim].start.subs({GPR_NUM: 0}) + offset[j], 1, 1
                    )
                    for j, dim in enumerate(custom.register_type.symbolic_shape)
                }
                ops_to_combine.append(write)

        # Useful to handle write/read dependency
        custom.replace_all_uses_with(ops_to_combine)
        custom.graph.erase_node(operator)


def partition_ops_with_gpr_offsets(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of reads and writes in a graph.
    If the reads or writes have incontiguous offsets based on GPR_NUM, we'd
    need to split these reads/writes appropriately.

    e.g a vector<16xf16> may be owned by lane 0, and lane 16 in this layout:
    [0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 16, 16, 16, 16].

    With our current glossary, this means we have 2 VGPR "chunks".
    [0:4) and [8:12) for lane0, and [4:8) and [12:16) for lane16.
    To the lane it should just look like vector<8xf16>.
    Hence for this example, we'd need two reads of vector<4xf16> and a couple
    insert_slices to combine them together into a single vector<8xf16>.

    """

    def has_gpr_offsets(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if not isinstance(custom, (Read, Write)):
            return False
        num_dims_with_gpr = sum(
            1 for v in custom.index.values() if v.start.has(GPR_NUM)
        )
        if num_dims_with_gpr == 1:
            return True
        elif num_dims_with_gpr == 0:
            return False
        raise NotImplementedError("Currently only handles 1 dim with GPR offset.")

    strided_operators = trace.walk(has_gpr_offsets)
    for operator in strided_operators:
        custom = get_custom(operator)
        simplified_index = {
            dim: simplify_index(custom.index.get(dim, custom.index[dim]))
            for dim in custom.index
        }
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        gpr_offsets = [
            v.start for v in simplified_index.values() if v.start.has(GPR_NUM)
        ]
        assert len(gpr_offsets) == 1, "Expected only 1-Dim has gpr offsets"
        gpr_offset_expr = gpr_offsets[0]
        gpr_cur_base_offset = gpr_offset_expr.subs({GPR_NUM: 0})
        cur_elem_id = 0
        with custom.graph.inserting_before(operator):
            ops_to_combine = []
            for i in range(elements_per_thread):
                # Break apart Reads/Writes that has non-contiguous GPR Read/Writes.
                next_gpr_offset = gpr_offset_expr.subs({GPR_NUM: i + 1})
                cur_gpr_offset = gpr_offset_expr.subs({GPR_NUM: i})
                gpr_offset_step = next_gpr_offset - cur_gpr_offset
                if not gpr_offset_step.is_number:
                    raise NotImplementedError(
                        "Only constant integer GPR offset steps supported."
                    )
                gpr_offset_step = int(gpr_offset_step)

                # Create new write when there is a jump in GPR offset
                # or at the end of the loop.
                if gpr_offset_step > 1 or i == elements_per_thread - 1:
                    # Get VGPR number of elements.
                    gpr_size = (cur_gpr_offset - gpr_cur_base_offset) + 1
                    assert gpr_size.is_number, "Expected gpr_size to be int."
                    gpr_size = int(gpr_size)

                    # Get updated index with VGPR offset.
                    updated_index_with_gpr_offset = {
                        dim: IndexSequence(
                            simplified_index[dim].start.subs({GPR_NUM: cur_elem_id}),
                            gpr_size,
                            simplified_index[dim].stride,
                        )
                        for dim in simplified_index
                    }

                    # Generate new Read/Write that has contiguous VGPR elements.
                    if isinstance(custom, Write):
                        extract = ExtractSlice(
                            custom.register_, [cur_elem_id], [gpr_size], [1]
                        ).add_to_graph(custom.graph)
                        new_node = Write(
                            extract,
                            custom.memory,
                            mapping=custom.mapping,
                            elements_per_thread=gpr_size,
                        ).add_to_graph(custom.graph)
                    elif isinstance(custom, Read):
                        # TODO: Add support on how to handle strided reads.
                        new_node = Read(
                            custom.memory,
                            elements_per_thread=gpr_size,
                            mapping=custom.mapping,
                            _write_dependency=custom._write_dependency,
                        ).add_to_graph(custom.graph)

                    # Update new_node information
                    new_node.index = updated_index_with_gpr_offset
                    new_node.vector_shapes = custom.vector_shapes
                    ops_to_combine.append(new_node)

                    # Set new current base GPR offset
                    gpr_cur_base_offset = next_gpr_offset
                    cur_elem_id = i + 1

            # Update users of original op.
            if isinstance(custom, Write):
                # Useful to handle write/read dependency
                custom.replace_all_uses_with(ops_to_combine)
            elif isinstance(custom, Read):
                reshape = Reshape(ops_to_combine, custom.vector_shapes).add_to_graph(
                    custom.graph
                )
                custom.replace_all_uses_with(reshape)
            custom.graph.erase_node(custom.fx_node)


def preprocess_nodes(
    constraints: Sequence[Constraint],
    mma_index: dict[MMA, dict[IndexSymbol, int]],
    mma_slices: dict[MMA, dict[IndexSymbol, list[fx.Node]]],
    node: fx.Node,
):
    set_vector_shapes(constraints, mma_index, mma_slices, node)
    set_node_index(constraints, mma_index, mma_slices, node)


def set_node_indices(trace: CapturedTrace, constraints: list[Constraint]):
    mma_index, mma_slices = get_mma_dimensional_mapping(
        trace, get_hardware_constraint(constraints)
    )
    trace.walk(partial(preprocess_nodes, constraints, mma_index, mma_slices))


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


def set_vector_shapes(
    constraints: Sequence[Constraint],
    mma_index: dict[MMA, dict[IndexSymbol, int]],
    mma_slices: dict[MMA, dict[IndexSymbol, list[fx.Node]]],
    node: fx.Node,
):
    """
    Set the vector shapes for the specific op based on whether the op lies in
    an MMA slice as well as the anchor node.
    """
    custom = get_custom(node)
    # MMA, Reduction & Reshape nodes already have their vector shapes set.
    if isinstance(custom, (MMA, Reduction, Reshape)):
        return
    # Add vector shapes from constraints to all ops. These are global constraints.
    custom.vector_shapes = {}
    hw_constraint = get_hardware_constraint(constraints)
    if hw_constraint.vector_shapes:
        custom.vector_shapes = hw_constraint.vector_shapes

    if len(mma_slices) == 1:
        # If there is just one MMA slice, there is no ambiguity in the vector shapes
        # and we set that singular MMA op as the anchor for all ops.
        mma = list(mma_slices.keys())[0]
        custom.anchor = mma
        custom.vector_shapes = custom.vector_shapes | mma.vector_shapes
        return

    for mma in mma_slices:
        if (
            node in mma_slices[mma][MMA_ACC]
            or node in mma_slices[mma][MMA_LHS]
            or node in mma_slices[mma][MMA_RHS]
        ):
            # Ensure that the operators indexing dims are present in the anchor.
            # For example, say we have a write node with indexing dimensions [B, M, N]
            # and there are two potential options for anchors: an MMA with
            # indexing dimensions [B, M, K1, K2] and another with indexing dimensions
            # [B, M, N, K2], we want to pick the second one otherwise the index
            # that is set from the anchor will not be accurate.
            if not set(custom.indexing_dims).issubset(mma.indexing_dims):
                continue
            custom.anchor = mma
            custom.vector_shapes = custom.vector_shapes | mma.vector_shapes
            return


def set_node_index(
    constraints: Sequence[Constraint],
    mma_index: dict[MMA, dict[IndexSymbol, int]],
    mma_slices: dict[MMA, dict[IndexSymbol, list[fx.Node]]],
    node: fx.Node,
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
    custom = get_custom(node)
    anchor = custom.anchor
    if isinstance(custom, (Reduction, Placeholder)) and not isinstance(custom, IterArg):
        return

    hardware_constraint = [get_hardware_constraint(constraints)]
    workgroup_constraints = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }
    other_constraints = [
        c for c in constraints if not isinstance(c, (HardwareConstraint, Assumption))
    ]
    # Apply hardware constraint first since it dictates the stride and size.
    sorted_constraints = hardware_constraint + other_constraints

    index = {}
    # The semantics of elements_per_thread are that it represents the number of
    # elements that are loaded contiguously from memory.
    elements_per_thread = getattr(custom, "elements_per_thread", None)
    # For elementwise operations that do not have an elements per thread attribute,
    # look back to the backward slice to see if they can find an appropriate value.
    # TODO: Remove this once set_node_index is integrated with thread_shape_analysis.
    if elements_per_thread is None:
        backward_slice = capture_backward_slice(node)
        for bwd_node in backward_slice:
            custom_node = get_custom(bwd_node)
            elements_per_thread = getattr(custom_node, "elements_per_thread", None)
            if elements_per_thread:
                break

    for dim in custom.indexing_dims:
        index_seq = None
        for constraint in sorted_constraints:
            if isinstance(constraint, HardwareConstraint):
                inputs = None
                if anchor and dim in mma_index[anchor]:
                    inputs = (mma_index[anchor][dim], elements_per_thread, None)
                else:
                    # Assumes vector shapes are associated with workgroup dims.
                    if dim not in workgroup_constraints:
                        continue
                    assert (
                        dim in constraint.vector_shapes
                    ), f"Dimension {dim} not found in vector shapes"
                    if constraint.vector_shapes[dim] == 0:
                        continue
                    inputs = (
                        workgroup_constraints[dim].workgroup_dim,
                        (
                            1
                            if not is_contiguous_dim(
                                dim,
                                custom.indexing_dims,
                                constraint.vector_shapes,
                            )
                            else elements_per_thread
                        ),
                        compute_stride(
                            custom.indexing_dims, constraint.vector_shapes, dim
                        ),
                    )
                    if elements_per_thread is None:
                        # Here we end up with a situation where there will be no thread level
                        # dependence in the dimensional index.
                        # TODO: Evaluate if this is a valid case.
                        continue
                index_seq = constraint.apply(
                    dim, *inputs, anchor and dim in mma_index[anchor]
                )
                if anchor and dim in mma_index[anchor]:
                    index_seq = specialize_index_sequence(
                        index_seq, mma_slices[anchor], custom
                    )

            elif constraint.dim == dim:
                if index_seq is None:
                    index_seq = constraint.apply()
                else:
                    index_seq.start += constraint.apply().start

        if index_seq is not None:
            index.update({dim: index_seq})
        else:
            index.update({dim: IndexSequence(0, 1, 1)})

    custom.index = index


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
