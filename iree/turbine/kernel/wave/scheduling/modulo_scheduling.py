# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from ....support.logging import get_logger
from .graph_utils import (
    Edge,
    EdgeWeight,
    find_strongly_connected_components,
    find_cycles_in_scc,
    all_pairs_longest_paths,
    evaluate_all_pairs_longest_paths,
    topological_sort,
    topological_sort_nodes,
)
from typing import Callable
import numpy as np
import math

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
        edges: list[Edge],
        resources: list[int],
    ) -> None:
        self.graph = graph
        self.edges = edges
        self.resources = resources
        self.seed = 2024

    def get_edge(self, from_node: fx.Node, to_node: fx.Node) -> Edge:
        """
        Returns the edge between two nodes.
        """
        for edge in self.edges:
            if edge._from == from_node and edge._to == to_node:
                return edge
        return None

    def get_edges_from_scheduled_node(
        self, edges: list[tuple[fx.Node, fx.Node]], to_node: fx.Node
    ) -> list[tuple[fx.Node, fx.Node]]:
        """
        Returns the edges that originate from a scheduled node and end
        in the specified node from the list of provided edges.
        """
        return [
            (from_, to_node)
            for (from_, to_) in edges
            if to_ == to_node and from_ in self.schedule
        ]

    def get_edges_to_scheduled_node(
        self, edges: list[tuple[fx.Node, fx.Node]], from_node: fx.Node
    ) -> list[tuple[fx.Node, fx.Node]]:
        """
        Returns the edges that end in a scheduled node and originate
        from the specified node from the list of provided edges.
        """
        return [
            (from_node, to_)
            for (from_, to_) in edges
            if from_ == from_node and to_ in self.schedule
        ]

    def all_scc_scheduled(self, sccs: dict[fx.Node, list[fx.Node]]) -> bool:
        """
        Checks if all strongly connected components have been scheduled.
        """
        for scc in sccs.values():
            for node in scc:
                if node not in self.schedule:
                    return False
        return True

    def schedule_graph(self) -> tuple[dict[fx.Node, int], bool]:
        """
        Schedule the graph using the Modulo Scheduler.
        Returns a schedule which maps each node to a cycle.
        """
        sccs = find_strongly_connected_components(self.graph, self.seed)

        logger.debug(f"Found {len(sccs)} strongly connected components.")
        for leader, nodes in sccs.items():
            logger.debug(
                f"Leader: {leader} owns {nodes} with finishing times {[x.f for x in nodes]}."
            )

        self.e_prime = self.find_edges(
            lambda edge: edge.weight.iteration_difference == 0
        )

        # Initialize initiation interval.
        T0 = int(max(self.compute_resource_ii(), self.compute_recurrence_ii(sccs)))

        # Compute symbolic all pairs longest path.
        e_star_symbolic = all_pairs_longest_paths(self.graph, self.edges)

        # Generate the schedule.
        # TODO: Come up with a better heuristic on an upper bound for the initiation interval.
        T_max_range = 3 * T0
        success = False
        for T in range(T0, T0 + T_max_range):
            logger.debug(f"Trying initiation interval: {T}.")
            self.RT = np.zeros((T, len(self.resources)))
            self.e_star = evaluate_all_pairs_longest_paths(e_star_symbolic, T)
            logger.debug(f"All Pairs Longest Paths: {self.e_star}.")
            self.schedule: dict[fx.Node, int] = {}
            for _, scc in topological_sort(sccs).items():
                logger.debug(f"Scheduling SCC: {scc}.")
                s0 = {}
                for node in scc:
                    candidate_edges = self.get_edges_from_scheduled_node(
                        self.e_star, node
                    )
                    s0[node] = 0
                    if candidate_edges:
                        s0[node] = max(
                            self.e_star[(from_node, to_node)] + self.schedule[from_node]
                            for (from_node, to_node) in candidate_edges
                        )
                first = min(s0, key=s0.get)
                s0 = s0[first]
                for s in range(s0, s0 + T):
                    if self.scc_scheduled(self.RT, T, scc, first, s):
                        logger.debug(f"Scheduled SCC: {scc} at time slot: {s}.")
                        logger.debug(f"Current RRT:\n {self.RT}.")
                        break
                else:
                    # If the SCC cannot be scheduled, increase the initiation interval.
                    logger.debug(f"Failed to schedule SCC: {scc}.")
                    break
            if self.all_scc_scheduled(sccs):
                success = True
                logger.debug(
                    f"Successfully scheduled all SCCs with initiation interval: {T}."
                )
                break
        else:
            raise Exception("Failed to schedule the graph.")

        self._initiation_interval = T
        return self.schedule, success

    def scc_scheduled(
        self,
        RT: np.array,
        T: int,
        scc: list[fx.Node],
        first: int,
        s: int,
    ) -> bool:
        """
        Tries to schedule the strongly connected component at time slot s. The nodes
        in the scc are scheduled in topological order based on the edges in E'.
        """
        RT_prime = np.array(RT)
        if not self.node_scheduled(RT_prime, T, first, s):
            logger.debug(f"Failed to schedule first node: {first}.")
            return False
        for node in topological_sort_nodes(scc, self.e_prime, [first]):
            logger.debug(f"Trying to schedule node: {node}.")
            sl = max(
                [
                    self.schedule[from_node] + self.e_star[(from_node, to_node)]
                    for (from_node, to_node) in self.get_edges_from_scheduled_node(
                        self.e_star, node
                    )
                ]
            )
            su = min(
                [
                    self.schedule[to_node] - self.e_star[(from_node, to_node)]
                    for (from_node, to_node) in self.get_edges_to_scheduled_node(
                        self.e_star, node
                    )
                ]
            )
            logger.debug(f"Lower bound: {sl}, Upper bound: {su}.")
            for s in range(sl, min(su, sl + T - 1) + 1):
                if self.node_scheduled(RT_prime, T, node, s):
                    logger.debug(f"Scheduled node: {node} at time slot: {s}.")
                    break
            else:
                # If the node cannot be scheduled, increase the initiation interval.
                logger.debug(f"Failed to schedule node: {node}.")
                return False
        RT[:] = np.array(RT_prime)
        return True

    def node_scheduled(self, RT: np.array, T: int, node: fx.Node, s: int) -> bool:
        """
        Checks for possible resource conflicts in the steady-state.
        """
        RT_prime = np.array(RT)
        for i in range(node.rrt.shape[0]):
            RT_prime[(s + i) % T] += node.rrt[i]
        if np.all(RT_prime <= self.resources):
            logger.debug(f"Scheduled node: {node} at time slot: {s}.")
            self.schedule[node] = s
            RT[:] = np.array(RT_prime)
            return True
        return False

    def compute_resource_ii(self) -> int:
        """
        Compute the resource constrained initiation interval.
        """
        usage = np.zeros(len(self.resources))
        for node in self.graph.nodes:
            usage += np.sum(node.rrt, axis=0)
        usage /= self.resources
        logger.debug(f"Resource constrained initiation interval: {np.max(usage)}.")
        return np.max(usage)

    def compute_recurrence_ii(self, scc: dict[fx.Node, list[fx.Node]]) -> int:
        """
        Compute the recurrence constrained initiation interval.
        """
        cycles = find_cycles_in_scc(scc)
        rec_ii = -1
        for cycle in cycles:
            delay, iteration_delay = 0, 0
            for from_node, to_node in zip(cycle[:-1], cycle[1:]):
                edge = self.get_edge(from_node, to_node)
                if edge is None:
                    continue
                delay += edge.weight.delay
                iteration_delay += edge.weight.iteration_difference
            rec_ii = max(rec_ii, delay / iteration_delay)
        logger.debug(f"Recurrence constrained initiation interval: {rec_ii}.")
        return rec_ii

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
        return self._initiation_interval

    @property
    def resource_reservations(self) -> np.array:
        """
        Returns the resource reservations of the schedule.
        """
        return self.RT

    @property
    def num_stages(self) -> int:
        """
        Returns the number of stages in the kernel of the pipelined loop.
        """
        max_cycle = max([t for t in self.schedule.values()])
        return math.ceil(max_cycle / self.initiation_interval)
