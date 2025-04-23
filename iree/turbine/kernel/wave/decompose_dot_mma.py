# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from .constraints import (
    Constraint,
    HardwareConstraint,
    GenericDot,
)
import torch.fx as fx
from ..ops.wave_ops import get_custom, MMA, Add, Mul, Sum, CastOp
from copy import copy


def decompose_dot_mma(trace: CapturedTrace, constraints: list[Constraint]):
    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )

    def get_mma_type(mma_op: MMA) -> GenericDot:
        mma_type = mma_op.mma_type
        if mma_type is None:
            mma_type = hardware_constraint.mma_type

        return mma_type

    def is_dot_mma(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, MMA):
            return False

        mma_type = get_mma_type(custom)
        return isinstance(mma_type, GenericDot)

    mma_nodes = trace.walk(is_dot_mma)
    for node in mma_nodes:
        mma_op = get_custom(node)
        mma_type = get_mma_type(mma_op)
        if mma_type.out_vec_size != 1:
            raise ValueError("Only support dot product with output vector size 1")

        with mma_op.graph.inserting_before(mma_op.fx_node):
            lhs = mma_op.lhs
            rhs = mma_op.rhs
            acc = mma_op.acc

            dtype = acc.type.dtype
            lhs_index = copy(lhs.index)
            rhs_index = copy(rhs.index)
            lhs = CastOp(lhs, dtype).add_to_graph(mma_op.graph)
            rhs = CastOp(rhs, dtype).add_to_graph(mma_op.graph)
            lhs.index = copy(lhs_index)
            rhs.index = copy(rhs_index)

            k_sym = get_custom(lhs).indexing_dims[1]

            mul = Mul(lhs, rhs).add_to_graph(mma_op.graph)
            sum = Sum(mul, None, k_sym).add_to_graph(mma_op.graph)
            red = Add(sum, acc).add_to_graph(mma_op.graph)

            mul.index = lhs_index | rhs_index
            del lhs_index[k_sym]
            del rhs_index[k_sym]
            ret_index = lhs_index | rhs_index
            sum.index = ret_index
            red.index = ret_index

            vector_shapes = mma_op.vector_shapes
            mul.vector_shapes = vector_shapes
            sum.vector_shapes = vector_shapes
            red.vector_shapes = vector_shapes

            mma_op.replace_all_uses_with(red)
