# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.hoisting")


def get_allocs(graph: fx.Graph) -> list[CustomOp]:
    return [
        custom_node
        for node in graph.nodes
        if isinstance((custom_node := get_custom(node)), Allocate)
    ]


def insert_broadcast(trace: CapturedTrace):
    """Insert broadcasts to binary ops operands that requires it."""

    def is_binary_op(node: fx.Node) -> bool:
        return isinstance(get_custom(node), BinaryPyOp)

    binary_nodes = trace.walk(is_binary_op)

    for node in binary_nodes:
        custom_node = get_custom(node)
        lhs = get_custom(custom_node.lhs)
        rhs = get_custom(custom_node.rhs)
        lhs_dim_set = set(lhs.type.symbolic_shape)
        rhs_dim_set = set(rhs.type.symbolic_shape)
        if lhs_dim_set == rhs_dim_set:
            continue
        if lhs_dim_set.isdisjoint(rhs_dim_set):
            raise ValueError("Cannot broadcast if lhs and rhs has disjointed shapes.")
        target_shape = lhs.type if lhs_dim_set > rhs_dim_set else rhs.type
        broadcast_idx, broadcast_src = (
            (1, rhs) if lhs_dim_set > rhs_dim_set else (0, lhs)
        )
        broadcast = Broadcast(broadcast_src.fx_node, target_shape)
        with custom_node.graph.inserting_before(custom_node.fx_node):
            broadcast.add_to_graph(custom_node.graph)
        custom_node.update_arg(broadcast_idx, broadcast.fx_node)
