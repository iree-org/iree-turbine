# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ops.wave_ops import Write, ExtractSlice, get_custom, Reduction
from .constraints import Constraint, HardwareConstraint, WorkgroupConstraint
from .._support.tracing import CapturedTrace, IndexingContext
from .._support.indexing import IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from .utils import (
    simplify_index,
    get_mma_dimensional_mapping,
    get_hardware_constraint,
    subs_idxc,
    specialize_index_sequence,
)
import torch.fx as fx
import numpy as np
from functools import partial
from typing import Sequence
from ...support.logging import get_logger
import sympy

logger = get_logger("turbine.wave.index_sequence_analysis")


def get_vector_shape(
    hardware_constraint: HardwareConstraint,
    symbolic_shape: list[IndexSymbol],
) -> list[int]:
    assert all(
        dim in hardware_constraint.vector_shapes for dim in symbolic_shape
    ), "Missing vector shape in hardware constraint"
    vector_shapes = [
        max(hardware_constraint.vector_shapes[dim], 1) for dim in symbolic_shape
    ]
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
            strides = [
                simplify_index(custom.register_index[dim]).stride
                for dim in custom.register_index
            ]
            elements_per_thread = [
                simplify_index(custom.register_index[dim]).size
                for dim in custom.register_index
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
            dim: simplify_index(custom.register_index.get(dim, custom.index[dim]))
            for dim in custom.index
        }

        shape = get_vector_shape(hw_constraint, custom.register_type.symbolic_shape)
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        max_stride_dim, max_stride = max(
            [(dim, seq.stride) for dim, seq in simplified_index.items()],
            key=lambda item: item[1],
        )
        with custom.graph.inserting_before(operator):
            for i in range(elements_per_thread):
                # Non-contiguous access patterns can have varying offsets. We
                # handle that here.
                gpr_offset = [
                    expr
                    for expr in simplified_index[max_stride_dim].start.args
                    if expr.has(GPR_NUM)
                ]
                if not gpr_offset:
                    gpr_offset = i
                else:
                    gpr_offset = sympy.Add(*gpr_offset).subs({GPR_NUM: i})
                extract = ExtractSlice(custom.register_, [i], [1], [1]).add_to_graph(
                    custom.graph
                )
                offset = np.unravel_index(int(gpr_offset * max_stride), shape)
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

        custom.graph.erase_node(operator)


def set_node_indices(trace: CapturedTrace, constraints: list[Constraint]):
    mma_index, mma_slices = get_mma_dimensional_mapping(
        trace, get_hardware_constraint(constraints)
    )
    trace.walk(partial(set_node_index, constraints, mma_index, mma_slices))


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


def set_node_index(
    constraints: Sequence[Constraint],
    mma_index: dict[IndexSymbol, int],
    mma_slices: dict[IndexSymbol, list[fx.Node]],
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
    hardware_constraint = [get_hardware_constraint(constraints)]
    workgroup_constraints = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }
    other_constraints = [
        c for c in constraints if not isinstance(c, HardwareConstraint)
    ]
    # Apply hardware constraint first since it dictates the stride and size.
    sorted_constraints = hardware_constraint + other_constraints

    index = {}
    # The semantics of elements_per_thread are that it represents the number of
    # elements that are loaded contiguously from memory.
    custom = get_custom(node)

    elements_per_thread = getattr(custom, "elements_per_thread", None)

    if isinstance(custom, Reduction):
        return

    for dim in custom.indexing_dims:
        index_seq = None
        for constraint in sorted_constraints:
            if isinstance(constraint, HardwareConstraint):
                inputs = None
                if dim in mma_index:
                    inputs = (mma_index[dim], elements_per_thread, None)
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
                index_seq = constraint.apply(dim, *inputs, dim in mma_index)
                if dim in mma_index:
                    index_seq = specialize_index_sequence(index_seq, mma_slices, custom)

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
    hw_cons = get_hardware_constraint(constraints)

    def apply_offset(node: fx.Node):
        custom = get_custom(node)
        if custom.expanded_dims is None:
            return False
        for dim, scale in custom.expanded_dims.items():
            if dim in custom.index:
                custom.index[dim].start += scale * hw_cons.vector_shapes[dim]
        return False

    trace.walk(apply_offset)
