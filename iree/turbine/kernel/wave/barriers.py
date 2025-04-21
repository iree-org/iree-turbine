# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils.graph_utils import is_reduction_subgraph
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, Read, SharedMemoryBarrier, Write, Iterate
from ..lang.global_symbols import SHARED_ADDRESS_SPACE
import torch.fx as fx
from typing import Optional


def add_shared_memory_barriers(
    trace: CapturedTrace,
    graph: Optional[fx.Graph] = None,
    last_node: Optional[fx.Node] = None,
) -> fx.Node:
    """
    Adds shared memory barriers to the graph. The barriers are inserted
    following a simple heuristic:
    - Read and write operations need a barrier between them.
    So we walk through the graph keeping track of the last read or write,
    and inserting a barrier before the next write or read.
    While sub-optimal, we use this as a baseline to compare more
    sophisticated barrier insertion strategies.
    """
    if not graph:
        graph = trace.get_root_graph()

    for node in graph.nodes:
        custom = get_custom(node)
        if (
            isinstance(custom, (Read, Write))
            and custom.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            if last_node is None:
                last_node = custom
                continue
            if type(custom) != type(last_node):
                # Synchronize after the write to shared memory before we read from it.
                with graph.inserting_before(node):
                    SharedMemoryBarrier().add_to_graph(graph)
            last_node = custom
        if isinstance(custom, Iterate):
            last_node = add_shared_memory_barriers(
                trace, trace.get_subgraph(custom.subgraph_name), last_node
            )

    # Synchronize before the write to shared memory to avoid stepping over
    # reads in the previous iteration of a loop.
    if is_reduction_subgraph(graph) and last_node:
        # Insert barrier at start of block, if load from shared memory exist.
        with graph.inserting_after(graph._root):
            SharedMemoryBarrier().add_to_graph(graph)

    return last_node
