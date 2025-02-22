# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from iree.turbine.kernel._support.tracing import CapturedTrace
from sympy import Symbol
from math import prod
from functools import cmp_to_key
from dataclasses import dataclass

import torch.fx as fx

from ..ops.wave_ops import get_custom, Allocate, SharedMemoryBarrier
from .constraints import Constraint, WorkgroupConstraint, TilingConstraint
from .utils import get_users, subs_idxc


def _get_location_vector(node: fx.Node) -> list[fx.Node]:
    location_vector = [node]
    cur_graph = node.graph
    while not hasattr(cur_graph, "subgraphs"):
        if not hasattr(cur_graph, "parent_op"):
            raise ValueError("All subgraphs should have parent_op")
        location_vector.append(cur_graph.parent_op)
        cur_graph = cur_graph.parent_op.graph
    return location_vector[::-1]


def _is_dominant_location_vector(
    src_vector: list[fx.node], other_vector: list[fx.Node]
):
    rank = min(len(src_vector), len(other_vector))
    for i in range(rank):
        if src_vector[i] == other_vector[i]:
            continue
        assert src_vector[i].graph == other_vector[i].graph
        return src_vector[i] > other_vector[i]


def _is_dominant_location(src_node: fx.Node, other_node: fx.Node):
    src_node_loc = _get_location_vector(src_node)
    other_node_loc = _get_location_vector(other_node)
    return _is_dominant_location_vector(src_node_loc, other_node_loc)


def _get_shmem_size(symbolic_shape: tuple[Symbol], constraints: list[Constraint]):
    shmem_size = []
    for dim in symbolic_shape:
        dim_tile_size = None
        for c in constraints:
            if not isinstance(c, WorkgroupConstraint) and not isinstance(
                c, TilingConstraint
            ):
                continue
            if c.dim != dim:
                continue
            dim_tile_size = subs_idxc(c.tile_size)
        # If not tiling info found, dim is not tiled so use actual size.
        dim_tile_size = dim_tile_size or subs_idxc(dim)
        assert dim_tile_size.is_integer, "tile size expected to be integer."
        shmem_size.append(int(dim_tile_size))
    assert len(shmem_size) == len(
        symbolic_shape
    ), "Failed to find tile size for one or more dims on shared alloc."
    return prod(shmem_size)


def reuse_shared_allocs(trace: CapturedTrace, constraints: list[Constraint]):
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
        shared_alloc_size = _get_shmem_size(alloc_type.symbolic_shape, constraints)
        candidates = allocs[(shared_alloc_size, alloc_type.dtype)]
        if not candidates or not _try_replace(node, candidates):
            candidates.append(node)


# TODO: wave viewOp -> memref.view
# TODO: better is_dead checker that is hoisting invariant
def _is_dead(current_node: fx.Node, node: fx.Node) -> bool:
    """
    Check if alloc `node` is dead, i.e. all uses are before `current_node`.
    """
    graph = current_node.graph
    current_users, _ = get_users(current_node, None)
    current_users = sorted(current_users, key=cmp_to_key(_is_dominant_location))
    users, _ = get_users(node, None)
    users = sorted(users, key=cmp_to_key(_is_dominant_location))

    current_start = current_users[0]
    current_end = current_users[-1]

    users_start = users[0]
    users_end = users[-1]

    return not (
        _is_dominant_location(users_end, current_start)
        and _is_dominant_location(current_end, users_start)
    )


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
