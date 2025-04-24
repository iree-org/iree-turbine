# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from ..._support.tracing import CapturedTrace
from ..._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from ...lang.global_symbols import *
from ...ops.wave_ops import (
    CustomOp,
    Reshape,
    MMA,
    get_custom,
)
from ..constraints import (
    HardwareConstraint,
    MMAType,
    MMAOperand,
)
import torch.fx as fx

from .symbol_utils import subs_idxc
from .graph_utils import capture_backward_slice


def is_reshape_needed(
    node: CustomOp,
    node_vector_shapes: dict[IndexSymbol, int],
    vector_shapes: dict[IndexSymbol, int],
) -> bool:
    for dim in node.type.symbolic_shape:
        if dim not in vector_shapes:
            # Ignore nodes that are not used in both mmas.
            return False
        if node_vector_shapes[dim] != vector_shapes[dim]:
            return True
    return False


def get_mma_dimensional_mapping(
    trace: CapturedTrace,
    hardware_constraint: HardwareConstraint,
) -> tuple[
    dict[MMA, dict[IndexSymbol, int]], dict[MMA, dict[IndexSymbol, list[fx.Node]]]
]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2). We maintain this map per mma node and
    also update the vector_shapes of the mma node based on this information.
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMA)

    mapping: dict[MMA, dict[IndexSymbol, int]] = {}
    mma_nodes = trace.walk(is_mma)
    for node in mma_nodes:
        custom: MMA = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape

        try:
            k = ((set(lhs_shape) & set(rhs_shape)) - set(acc_shape)).pop()
        except KeyError as e:
            raise RuntimeError(
                f"{node}: Invalid MMA shapes\n{lhs_shape=}\n{rhs_shape=}\n{acc_shape=}\n{m=}, {n=}\n{custom}"
            )
        if m not in lhs_shape or n not in rhs_shape:
            raise RuntimeError(
                f"{node}: Invalid MMA shapes\n{lhs_shape=}\n{rhs_shape=}\n{acc_shape=}\n{m=}, {n=}, {k=}\n{custom}"
            )

        if custom not in mapping:
            mapping[custom] = {}
        mapping[custom][m] = MMAOperand.M
        mapping[custom][n] = MMAOperand.N
        mapping[custom][k] = MMAOperand.K
        custom.vector_shapes = {
            m: hardware_constraint.mma_matrix_shapes(custom.mma_type)[0],
            n: hardware_constraint.mma_matrix_shapes(custom.mma_type)[1],
            k: hardware_constraint.mma_matrix_shapes(custom.mma_type)[2],
        }
        if hardware_constraint.vector_shapes:
            custom.vector_shapes.update(hardware_constraint.vector_shapes)
        custom.reduction_dim = k

        # Since expansion proceeds bottom-up, we set the vector shapes
        # of the parent reduction to the vector shapes of the last MMA node.
        if hasattr(custom.graph, "parent_op"):
            reduction = get_custom(custom.graph.parent_op)
            reduction.vector_shapes = custom.vector_shapes

    # Determine if any reshapes are required. Reshapes are added for
    # chained matmuls when the vector shapes of the operands in one matmul
    # differ from those in another matmul. The mma_slices contain all the ops
    # in the backward slice of the lhs and rhs upto a previous mma (if one exists).
    # So we check for the previous node of the first operator in the slice to see
    # if it is an MMA and if so check if a reshape is required.
    def add_reshape_if_needed(mma: MMA, prev_mma: MMA, arg_index: int):
        with mma.graph.inserting_before(mma.fx_node):
            arg = mma.lhs if arg_index == 0 else mma.rhs
            arg = get_custom(arg)
            if is_reshape_needed(arg, mma.vector_shapes, prev_mma.vector_shapes):
                reshape = Reshape(arg.fx_node, prev_mma.vector_shapes).add_to_graph(
                    mma.graph
                )
                custom_reshape = get_custom(reshape)
                custom_reshape.vector_shapes = mma.vector_shapes
                mma.update_arg(arg_index, reshape)

    def find_mma_in_slice(node: CustomOp) -> Optional[MMA]:
        """
        Find the closest mma by iterating through the backward slice of a node
        in reverse.
        """
        slice = list(capture_backward_slice(node))
        for arg in reversed(slice):
            prev_mma = get_custom(arg)
            if isinstance(prev_mma, MMA):
                return prev_mma
        return None

    # Look in the backward slices of both the LHS and RHS to find
    # mmas. If found, add reshapes if necessary.
    for mma in mma_nodes:
        custom_mma = get_custom(mma)
        prev_mma = find_mma_in_slice(custom_mma.lhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 0)
        prev_mma = find_mma_in_slice(custom_mma.rhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 1)

    return mapping


def get_mfma_load_elems_per_thread(mfma_variant: MMAType) -> int:
    match mfma_variant:
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 4
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 8
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K8_F16
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 8


def get_mfma_store_elems_per_thread(mfma_variant: MMAType) -> int:
    match mfma_variant:
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 16
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 4
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_K8_F16
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 16


def simplify_index(index: IndexExpr) -> IndexExpr:
    """
    Simplifies the index by applying the following bindings:
        - MMA acc_index bindings so the index of the MMA node is the acc_index.
    """
    if isinstance(index, int):
        return index
    mapping = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
    return subs_idxc(index.subs(mapping))


def specialize_index_sequence(
    index_seq: IndexSequence,
    mma_slices: dict[IndexSymbol, list[fx.Node]],
    custom: CustomOp,
) -> IndexSequence:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    If the node is not used as any of the operands, return the original index sequence
    with all the MMA symbols zeroed out.
    """
    if isinstance(custom, MMA):
        return index_seq
    operand_map = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 0}
    for key in mma_slices:
        if custom.fx_node in mma_slices[key]:
            operand_map[key] = 1
            return index_seq.subs(operand_map)
    return index_seq.subs(operand_map)
