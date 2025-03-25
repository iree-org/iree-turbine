# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    MMAType,
    MMAOperand,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import (
    get_custom,
    MMA,
    Reshape,
)
from ..lang.global_symbols import *

import copy
import sympy


VMMA_TO_NATIVE_MAP = {
    MMAType.F32_16x16x32_K8_F16: MMAType.F32_16x16x16_F16,
    MMAType.F32_32x32x16_K8_F16: MMAType.F32_32x32x8_F16,
}


def get_m_dim(mma_op, vmma_expr):
    m_sets = set(mma_op.lhs_type.symbolic_shape).difference(
        mma_op.rhs_type.symbolic_shape
    )
    # Filters candidate for M-dims. Adds candidate dim to list iff we find any occurrence
    # of the reference MMA template expression in the candidate dim's index expression
    vmma_expr = vmma_expr.subs({MMA_ACC: 0})
    m_dims = set(
        [
            m_candidate
            for m_candidate in m_sets
            if mma_op.lhs.index[m_candidate].start.find(vmma_expr)
        ]
    )
    assert len(m_dims) == 1, "Expect to have single M-dim."
    return m_dims.pop()


def get_n_dim(mma_op, vmma_expr):
    n_sets = set(mma_op.rhs_type.symbolic_shape).difference(
        mma_op.lhs_type.symbolic_shape
    )
    # Filters candidate for N-dims. Adds candidate dim to list iff we find any occurrence
    # of the reference MMA template expression in the candidate dim's index expression
    vmma_expr = vmma_expr.subs({MMA_ACC: 0})
    n_dims = set(
        [
            n_candidate
            for n_candidate in n_sets
            if mma_op.rhs.index[n_candidate].start.find(vmma_expr)
        ]
    )
    assert len(n_dims) == 1, "Expect to have single N-dim."
    return n_dims.pop()


def get_k_dim(mma_op, vmma_expr):
    k_dims = set(mma_op.lhs_type.symbolic_shape).intersection(
        mma_op.rhs_type.symbolic_shape
    )
    k_dims = k_dims.difference(mma_op.acc_type.symbolic_shape)
    assert len(k_dims) == 1, "Expect to have single K-dim."
    return k_dims.pop()


def replace_subexpr(
    main_expr: sympy.Expr, src_subexpr: sympy.Expr, dst_subexpr: sympy.Expr
):
    assert (
        len(main_expr.find(src_subexpr)) > 0
    ), "Expected src subexpression to be found in main expression."
    # Early exit if src and dst is already the same.
    if src_subexpr == dst_subexpr:
        return
    main_expr = main_expr.subs(src_subexpr, dst_subexpr)


def decompose_vmma_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    The decomposition for VMMA is done in five steps:
      1. Collect all MMA ops and only process the ones with virtual intrinsics.
      2. For each VirtualMMAOp, we generate check it's unroll factor
      3. Generate slices of lhs and rhs from coaleseced reads based on native and virtual MMA sizes
      4. Generate native MMA ops based on unroll factor that takes in slices of coaleseced lhs and rhs
      5. Modify index expression of new mma op by replacing the expression based on virtual layout
         with it's equivalent native layout expression.
    """
    # Get MMA nodes.
    mma_nodes = trace.walk(lambda node: isinstance(get_custom(node), MMA))
    if not mma_nodes:
        return
    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    for node in mma_nodes:
        mma_op = get_custom(node)
        mma_type = mma_op.mma_type or hardware_constraint.mma_type
        # Only process VirtualMMAOps.
        if mma_type not in VMMA_TO_NATIVE_MAP:
            continue
        native_mma_type = VMMA_TO_NATIVE_MAP[mma_type]
        virtual_vector_shapes = mma_op.vector_shapes
        native_vector_shapes = {
            k: v
            for k, v in zip(
                virtual_vector_shapes,
                hardware_constraint.mma_matrix_shapes(native_mma_type),
            )
        }

        innermost_dim = mma_op.reduction_dim

        unrollKFactor = (
            virtual_vector_shapes[innermost_dim] // native_vector_shapes[innermost_dim]
        )
        assert unrollKFactor > 1, "Expected Unroll K factor to be > 1"

        mma_acc = mma_op.acc
        for i in range(unrollKFactor):
            # Emitting ReshapeOp for slicing
            with mma_op.graph.inserting_before(mma_op.fx_node):
                slice_lhs = Reshape([mma_op.lhs], virtual_vector_shapes).add_to_graph(
                    mma_op.graph
                )
                slice_rhs = Reshape([mma_op.rhs], virtual_vector_shapes).add_to_graph(
                    mma_op.graph
                )

            # Setting vector_shapes for num_partitions
            slice_lhs.vector_shapes = native_vector_shapes
            slice_rhs.vector_shapes = native_vector_shapes

            # Setting offset
            slice_lhs.expanded_dims = {innermost_dim: i}
            slice_rhs.expanded_dims = {innermost_dim: i}

            # Generate new MMA
            vmma_expr = hardware_constraint.mma_index_offset(mma_type)
            mma_expr = hardware_constraint.mma_index_offset(native_mma_type)
            with mma_op.graph.inserting_before(mma_op.fx_node):
                mma_acc = MMA(
                    slice_lhs, slice_rhs, mma_acc, native_mma_type
                ).add_to_graph(mma_op.graph)
                mma_acc.index = copy.deepcopy(mma_op.index)
                m_dim = get_m_dim(mma_op, vmma_expr[MMAOperand.M.value])
                n_dim = get_n_dim(mma_op, vmma_expr[MMAOperand.N.value])
                k_dim = get_k_dim(mma_op, vmma_expr[MMAOperand.K.value])
                # Replace expression based on virtual with it's equivalence in the native layout.
                replace_subexpr(
                    mma_acc.index[m_dim].start,
                    vmma_expr[MMAOperand.M.value],
                    mma_expr[MMAOperand.M.value],
                )
                replace_subexpr(
                    mma_acc.index[n_dim].start,
                    vmma_expr[MMAOperand.N.value],
                    mma_expr[MMAOperand.N.value],
                )
                replace_subexpr(
                    mma_acc.index[k_dim].start,
                    vmma_expr[MMAOperand.K.value],
                    mma_expr[MMAOperand.K.value],
                )
        mma_op.replace_all_uses_with(mma_acc)
