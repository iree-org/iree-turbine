from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    get_custom,
    Read,
    SharedMemoryBarrier,
    Write,
    CustomOp,
    Reduction,
)
import torch.fx as fx
from ..lang.global_symbols import SHARED_ADDRESS_SPACE


def add_shared_memory_barriers(
    trace: CapturedTrace | fx.Graph, last_node: CustomOp = None
):
    """
    Adds shared memory barriers to the graph. The barriers are inserted
    by detecting read-after-write (RAW) based on the write dependencies
    of the read nodes and the access patterns of the read and write
    nodes. To minimize the number of barriers, we only insert barriers
    between nodes if there are no existing barriers between them.
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
            add_shared_memory_barriers(
                trace.get_subgraph(custom.subgraph_name), last_node
            )
