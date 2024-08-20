from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, Read, SharedMemoryBarrier
import torch.fx as fx


def add_shared_memory_barriers(trace: CapturedTrace):
    """
    Adds shared memory barriers to the graph. The barriers are inserted
    by detecting read-after-write (RAW) based on the write dependencies
    of the read nodes and the access patterns of the read and write
    nodes. To minimize the number of barriers, we only insert barriers
    between nodes if there are no existing barriers between them.
    """
    barriers: list[SharedMemoryBarrier] = []
    for subgraph in trace.region_graph.subgraphs.values():
        for node in subgraph.nodes:
            custom = get_custom(node)
            if isinstance(custom, Read) and custom.write_dependency:
                if custom.index != custom.write_dependency.index:
                    if any(
                        barrier.is_barrier_between(custom.write_dependency, node)
                        for barrier in barriers
                    ):
                        continue

                    with subgraph.inserting_before(node):
                        barrier = SharedMemoryBarrier().add_to_graph(subgraph)
                        barriers.append(get_custom(barrier))
