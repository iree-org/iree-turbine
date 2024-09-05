import torch.fx as fx
from enum import Enum
from ...ops.wave_ops import CustomOp, MMA
from ....support.logging import get_logger
from .graph_utils import find_strongly_connected_components, Edge, EdgeWeight
from typing import Callable
import numpy as np

logger = get_logger("turbine.wave.modulo_scheduling")


class ModuloScheduler:
    """
    Vanilla Modulo Scheduler.
    References:
    [1] Aho, Alfred V., et al. "Compilers: Principles, Techniques, and Tools."
    """

    def __init__(
        self,
        graph: fx.Graph,
        weighted_edges: dict[Edge, EdgeWeight],
        resources: list[int],
    ) -> None:
        self.graph = graph
        self.edges = weighted_edges
        self.resources = resources
        self.seed = 2024

    def schedule(self) -> dict[fx.Node, int]:
        """
        Schedule the graph using the Modulo Scheduler.
        Returns a schedule which maps each node to a cycle.
        """
        scc = find_strongly_connected_components(self.graph, self.seed)

        logger.debug(f"Found {len(scc)} strongly connected components.")
        for leader, nodes in scc.items():
            logger.debug(
                f"Leader: {leader} owns {nodes} with finishing times {[x.f for x in nodes]}."
            )

    def compute_resource_ii(self) -> int:
        """
        Compute the resource constrained initiation interval.
        """
        usage = np.zeros(len(self.resources))
        for node in self.graph.nodes:
            usage += np.sum(node.rrt, axis=0)
        usage /= self.resources
        return np.max(usage, axis=1)

    def compute_recurrence_ii(self) -> int:
        """
        Compute the recurrence constrained initiation interval.
        """
        pass

    def find_edges(self, filter: Callable[[Edge], bool]) -> list[Edge]:
        filtered = []
        for edge in self.edges:
            if filter(edge):
                filtered.append(edge)
        return filtered

    @property
    def initiation_interval(self) -> int:
        """
        Returns the initiation interval of the schedule.
        """
        pass
