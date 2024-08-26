from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, Read, SharedMemoryBarrier
import torch.fx as fx


def get_last_write_dependency(write_dependency: list[Read]) -> Read:
    """
    Given a list of write dependencies, returns the last write dependency
    in the graph. The last write dependency is the one that is
    furthest from the root/start of the graph.
    """
    if len(write_dependency) == 1:
        return write_dependency[0]
    graph = get_custom(write_dependency[0]).graph
    for node in reversed(graph.nodes):
        if node in write_dependency:
            return node


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
                write_dependency = get_last_write_dependency(custom.write_dependency)
                if any(
                    barrier.is_barrier_between(write_dependency, node)
                    for barrier in barriers
                ):
                    continue

                if (
                    len(custom.write_dependency) > 1
                    or custom.index != custom.write_dependency[0].index
                ):
                    with subgraph.inserting_before(node):
                        barrier = SharedMemoryBarrier().add_to_graph(subgraph)
                        barriers.append(get_custom(barrier))
