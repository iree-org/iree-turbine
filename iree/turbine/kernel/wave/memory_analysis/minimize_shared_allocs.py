# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from iree.turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
import numpy as np

from ...ops.wave_ops import get_custom, Allocate, SharedMemoryBarrier, NestedRegionOp
from ..utils.graph_utils import get_users
from ..utils.symbol_utils import subs_idxc
from dataclasses import dataclass
from ...lang.global_symbols import *
from ..._support.dtype import i8
from .solver import (
    determine_allocations_offsets,
)
import math
from ..utils.graph_utils import is_barrier_between


def update_sort_keys(
    trace: CapturedTrace, graph: fx.Graph, prefix: Optional[tuple] = ()
):
    """
    Update the sort keys of the graph so that
    consecutive nodes have consecutive sort keys.
    Also, broadcast the sort keys for ops in nested graphs.
    After this pass, the sort keys are unique and monotonically increasing.
    For example, if we have a graph with nodes [a, b, c, d], and c is a nested
    region with ops [e, f, g], then the sort keys will be:

    a: (0,)
    b: (1,)
    c: (2,)
        e: (2, 0)
        f: (2, 1)
        g: (2, 2)
    d: (3,)

    so that we can always say that a < b < c < e < f < g < d.
    """
    for i, node in enumerate(graph.nodes):
        node._sort_key = prefix + (i,)
        custom = get_custom(node)
        if isinstance(custom, NestedRegionOp):
            update_sort_keys(
                trace,
                trace.region_graph.subgraphs[custom.subgraph_name],
                node._sort_key,
            )


@dataclass
class LiveInterval:
    start: tuple[int] = (10000,)
    end: tuple[int] = (-1,)


def compute_live_intervals(allocs: list[fx.Node]):
    """
    Compute the live intervals for the allocs.
    """
    live_intervals = {}
    for alloc in allocs:
        live_intervals[alloc] = LiveInterval()
        users, _ = get_users(alloc, None)
        for user in users:
            if user._sort_key < live_intervals[alloc].start:
                live_intervals[alloc].start = user._sort_key
            if user._sort_key > live_intervals[alloc].end:
                live_intervals[alloc].end = user._sort_key
    return live_intervals


def get_shared_memory_allocation_size(alloc: fx.Node) -> tuple[int, int, int]:
    custom = get_custom(alloc)
    return math.prod([subs_idxc(x) for x in custom.distributed_shape])


def get_use(
    alloc: fx.Node, live_interval: LiveInterval, match_sort_key: int
) -> fx.Node:
    users, _ = get_users(alloc, None)
    matches = [x for x in users if x._sort_key == live_interval.start]
    if len(matches) != 1:
        raise ValueError(
            f"Expected 1 match for {alloc} and {match_sort_key}, got {len(matches)}"
        )
    return matches[0]


def get_first_use(alloc: fx.Node, live_interval: LiveInterval) -> fx.Node:
    return get_use(alloc, live_interval, live_interval.start)


def get_last_use(alloc: fx.Node, live_interval: LiveInterval) -> fx.Node:
    return get_use(alloc, live_interval, live_interval.end)


def insert_barrier_if_needed(alloc: fx.Node, first_use: fx.Node, last_use: fx.Node):
    """
    This function inserts a barrier between the last use of the allocation i
    and the first use of allocation j, if no barrier already exists between them.
    The barrier is inserted before the first use of allocation j, unless
    allocation i and allocation j are in different graphs in which case
    the barrier is inserted before the first use parent op.
    """
    if is_barrier_between(last_use, first_use):
        return

    if last_use.graph != first_use.graph:
        first_use = first_use.graph.parent_op
    with first_use.graph.inserting_before(first_use):
        SharedMemoryBarrier().add_to_graph(first_use.graph)


def minimize_shared_allocs(trace: CapturedTrace, minimize_shared_allocs: bool):
    """
    Minimize the number of shared allocs by reusing them.
    See: iree/turbine/docs/kernel/memory_analysis.md for more details.
    """
    if not minimize_shared_allocs:
        return
    update_sort_keys(trace, trace.get_root_graph())

    allocs = trace.walk(lambda x: isinstance(get_custom(x), Allocate))
    live_intervals = compute_live_intervals(allocs)

    alloc_info = [
        (
            get_shared_memory_allocation_size(x),
            live_intervals[x].start,
            live_intervals[x].end,
        )
        for x in allocs
    ]
    offsets, allocation_size = determine_allocations_offsets(alloc_info)
    if offsets is None:
        raise ValueError("No feasible solution found for shared memory allocation.")

    shared_memory = np.zeros(allocation_size, dtype=np.int8)
    time_sorted_allocs = sorted(allocs, key=lambda x: live_intervals[x].start)
    last_use = None
    for i, alloc in enumerate(time_sorted_allocs):
        offset = offsets[i]
        if np.any(
            shared_memory[offset : offset + get_shared_memory_allocation_size(alloc)]
        ):
            first_use = get_first_use(alloc, live_intervals[alloc])
            insert_barrier_if_needed(alloc, first_use, last_use)
        shared_memory[offset : offset + get_shared_memory_allocation_size(alloc)] = 1
        last_use = get_last_use(alloc, live_intervals[alloc])

    # Create a 1D parent allocation for each allocation.
    combined_shape = set()
    for alloc in allocs:
        combined_shape.update(get_custom(alloc).shape)

    first_node = list(trace.get_root_graph().nodes)[0]
    with trace.get_root_graph().inserting_before(first_node):
        parent = Allocate(
            shape=list(combined_shape),
            distributed_shape=[allocation_size],
            dtype=get_custom(allocs[0]).type.dtype,
            address_space=SHARED_ADDRESS_SPACE,
        )
        parent.add_to_graph(trace.get_root_graph())

    for alloc, offset in zip(allocs, offsets):
        get_custom(alloc).update_arg("parent", parent)
        get_custom(alloc).update_arg("offset", offset)

    return
