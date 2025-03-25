# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from iree.turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx

from ..ops.wave_ops import get_custom, Allocate, SharedMemoryBarrier
from .utils.graph_utils import get_users


def reuse_shared_allocs(trace: CapturedTrace):
    """
    Reuse shared allocs if their lifetimes doesn't intersect
    """
    allocs = defaultdict(list)

    # This pass is supposed to be run after hoisting so we only check top-level
    # graph.
    root_graph = trace.get_root_graph()
    for node in list(root_graph.nodes):
        alloc = get_custom(node)
        if not isinstance(alloc, Allocate):
            continue

        alloc_type = alloc.type
        candidates = allocs[(alloc_type.symbolic_shape, alloc_type.dtype)]
        if not candidates or not _try_replace(node, candidates):
            candidates.append(node)


def _is_dead(current_node: fx.Node, node: fx.Node) -> bool:
    """
    Check if alloc `node` is dead, i.e. all uses are before `current_node`.
    """
    graph = current_node.graph
    users, _ = get_users(node, None)
    for user in users:
        # If user is node in nested graph, get parent node.
        while user.graph != graph:
            user = user.graph.parent_op

        if user >= current_node:
            return False

    return True


def _try_replace(current_node: fx.Node, candidates: list[fx.Node]) -> bool:
    """
    Try to replace `current_node` if we have some dead alloc nodes
    """
    for candidate in candidates:
        if _is_dead(current_node, candidate):
            if not isinstance(get_custom(current_node.prev), SharedMemoryBarrier):
                graph = current_node.graph
                with graph.inserting_before(current_node):
                    SharedMemoryBarrier().add_to_graph(graph)

            custom = get_custom(current_node)
            custom.replace_all_uses_with(candidate)
            custom.erase()
            return True

    return False
