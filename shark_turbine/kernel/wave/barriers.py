# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, Read, SharedMemoryBarrier, Write, Reduction
from ..lang.global_symbols import SHARED_ADDRESS_SPACE
import torch.fx as fx


def add_shared_memory_barriers(
    trace: CapturedTrace | fx.Graph, last_node: fx.Node = None
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
    graph = trace
    if isinstance(trace, CapturedTrace):
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
                with graph.inserting_before(node):
                    SharedMemoryBarrier().add_to_graph(graph)
            last_node = custom
        if isinstance(custom, Reduction):
            last_node = add_shared_memory_barriers(
                trace.get_subgraph(custom.subgraph_name), last_node
            )

    return last_node
