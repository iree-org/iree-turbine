from typing import Dict, List, Optional, Tuple, Callable, NamedTuple
import numpy as np
import torch
import torch.fx as fx
from .resources import get_available_resources
from .graph_utils import Edge
from ...ops.wave_ops import get_custom, IterArg

Schedule = Dict[fx.Node, int]
RawEdgesList = Optional[List[Edge]]
NodeRRTGetter = Callable[[fx.Node], np.ndarray]


class ResourceUsageTracker:
    """Tracks and validates resource usage across scheduling cycles."""

    def __init__(self, resource_limits: np.ndarray, T: int, num_resource_types: int):
        self.resource_limits = np.array(resource_limits)
        self.T = T
        self.num_resource_types = num_resource_types
        self.RT_global = np.zeros((self.T, self.num_resource_types), dtype=int)

    def _get_node_duration(self, node: fx.Node, node_rrt_getter: NodeRRTGetter) -> int:
        node_rrt_val = node_rrt_getter(node)
        return (
            node_rrt_val.shape[0]
            if node_rrt_val is not None and node_rrt_val.size > 0
            else 0
        )

    def _apply_node_operation(
        self,
        node: fx.Node,
        start_cycle: int,
        node_rrt_getter: NodeRRTGetter,
        operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        node_rrt_val = node_rrt_getter(node)
        if node_rrt_val is None or node_rrt_val.size == 0:
            return
        for i in range(node_rrt_val.shape[0]):
            cycle = (start_cycle + i) % self.T
            self.RT_global[cycle, :] = operation(
                self.RT_global[cycle, :], node_rrt_val[i, :]
            )

    def add_node(
        self, node: fx.Node, start_cycle: int, node_rrt_getter: NodeRRTGetter
    ) -> None:
        self._apply_node_operation(
            node, start_cycle, node_rrt_getter, lambda x, y: x + y
        )

    def remove_node(
        self, node: fx.Node, start_cycle: int, node_rrt_getter: NodeRRTGetter
    ) -> None:
        self._apply_node_operation(
            node, start_cycle, node_rrt_getter, lambda x, y: x - y
        )

    def can_add_node(
        self, node: fx.Node, start_cycle: int, node_rrt_getter: NodeRRTGetter
    ) -> bool:
        node_rrt_val = node_rrt_getter(node)
        if node_rrt_val is None or node_rrt_val.size == 0:
            return True
        for i in range(node_rrt_val.shape[0]):
            cycle = (start_cycle + i) % self.T
            if np.any(
                self.RT_global[cycle, :] + node_rrt_val[i, :] > self.resource_limits
            ):
                return False
        return True


class ScheduleDependencyGraph:
    """Represents and manages the dependency relationships between scheduled operations."""

    def __init__(self, nodes: List[fx.Node], edges: RawEdgesList = None):
        self.nodes = list(nodes)
        self.edges = edges
        self._adj = self._build_adjacency_list(edges, is_successors=True)
        self._pred_adj = self._build_adjacency_list(edges, is_successors=False)

    def _build_adjacency_list(
        self, edges_input: RawEdgesList, is_successors: bool
    ) -> Dict[fx.Node, List[fx.Node]]:
        adj_map = {node: [] for node in self.nodes}
        for edge in edges_input:
            if is_successors:
                if edge._from in adj_map and edge._to not in adj_map[edge._from]:
                    adj_map[edge._from].append(edge._to)
            else:
                if edge._to in adj_map and edge._from not in adj_map[edge._to]:
                    adj_map[edge._to].append(edge._from)
        return adj_map

    def get_successors(self, node: fx.Node) -> List[fx.Node]:
        return self._adj.get(node, [])

    def get_predecessors(self, node: fx.Node) -> List[fx.Node]:
        return self._pred_adj.get(node, [])

    def has_edge(self, pred: fx.Node, succ: fx.Node) -> bool:
        return (
            any(edge._from == pred and edge._to == succ for edge in self.edges)
            if self.edges
            else False
        )


class ScheduleConstraintRepairer:
    """Repairs schedule violations by moving operations to satisfy resource and dependency constraints."""

    def __init__(self, graph: ScheduleDependencyGraph, edges: List[Edge], T: int):
        self.graph = graph
        self.edges = edges
        self.T = T
        self.MAX_REPAIR_ITERATIONS = len(graph.nodes)

    def _repair_schedule(
        self,
        schedule: Schedule,
        resource_tracker: ResourceUsageTracker,
        node_rrt_getter: NodeRRTGetter,
        forward: bool = True,
    ) -> Tuple[bool, Schedule]:
        """Repairs a schedule to satisfy dependency and resource constraints.

        The repair process iteratively moves nodes either forward or backward in time until:
        1. All constraints are satisfied
        2. No more valid moves can be made
        3. Maximum repair iterations are reached

        Uses a directional constraint enforcement strategy:
        - Forward repair (forward=True):
          * Process nodes in ascending order
          * Enforce predecessor constraints only
          * Handle successor violations when processing successors

        - Backward repair (forward=False):
          * Process nodes in descending order
          * Enforce successor constraints only
          * Handle predecessor violations when processing predecessors

        This strategy maintains schedule validity while avoiding unnecessary cascading repairs.

        Args:
            schedule: Current schedule to repair
            resource_tracker: Resource tracker to validate resource constraints
            node_rrt_getter: Function to get resource requirements for nodes
            forward: If True, repair by moving nodes forward; if False, repair by moving nodes backward

        Returns:
            Tuple of (success, repaired_schedule) where:
            - success: True if repair was successful, False if repair failed
            - repaired_schedule: The repaired schedule if successful, or the last attempted schedule if failed
        """
        repaired_schedule = dict(schedule)

        def get_constraint_cycles(node: fx.Node) -> Tuple[List[int], List[int]]:
            """Get cycles of predecessor and successor nodes that have edges to/from the current node."""
            pred_cycles = [
                repaired_schedule[pred]
                for pred in self.graph.get_predecessors(node)
                if pred in repaired_schedule and self.graph.has_edge(pred, node)
            ]
            succ_cycles = [
                repaired_schedule[succ]
                for succ in self.graph.get_successors(node)
                if succ in repaired_schedule and self.graph.has_edge(node, succ)
            ]
            return pred_cycles, succ_cycles

        def should_move_node(
            node: fx.Node, pred_cycles: List[int], succ_cycles: List[int]
        ) -> Tuple[bool, int]:
            """Determine if a node should be moved and calculate its target cycle."""
            if forward:
                # Forward repair: enforce predecessor constraints
                if pred_cycles and repaired_schedule[node] <= max(pred_cycles):
                    return True, max(pred_cycles) + 1
            else:
                # Backward repair: enforce successor constraints
                if succ_cycles and repaired_schedule[node] >= min(succ_cycles):
                    return True, min(succ_cycles) - 1
            return False, 0

        for _ in range(self.MAX_REPAIR_ITERATIONS):
            schedule_modified = False
            # Sort nodes based on direction (forward/backward)
            nodes_to_check = sorted(
                repaired_schedule.keys(),
                key=lambda n: (
                    repaired_schedule[n] if forward else -repaired_schedule[n]
                ),
            )

            for current_node in nodes_to_check:
                pred_cycles, succ_cycles = get_constraint_cycles(current_node)
                should_move, target_cycle = should_move_node(
                    current_node, pred_cycles, succ_cycles
                )

                if should_move:
                    if not self._try_move_node(
                        current_node,
                        target_cycle,
                        repaired_schedule,
                        resource_tracker,
                        node_rrt_getter,
                        forward,
                    ):
                        return False, repaired_schedule
                    schedule_modified = True

            if not schedule_modified:
                break

        return True, repaired_schedule

    def _try_move_node(
        self,
        node: fx.Node,
        target_cycle: int,
        schedule: Schedule,
        resource_tracker: ResourceUsageTracker,
        node_rrt_getter: NodeRRTGetter,
        forward: bool,
    ) -> bool:
        """Attempts to move a node to a target cycle while maintaining schedule validity.

        This method handles the actual movement of a node to a new cycle, including:
        1. Temporarily removing the node from its current cycle
        2. Finding a valid cycle to place the node
        3. Validating resource and dependency constraints
        4. Restoring the original state if the move fails

        For forward repair (forward=True):
        - Tries cycles from target_cycle up to target_cycle + T
        - Returns True on first valid cycle found
        - Returns False if no valid cycle is found

        For backward repair (forward=False):
        - Tries cycles from target_cycle down to target_cycle - T
        - Returns True on first valid cycle found
        - Returns False if no valid cycle is found

        Args:
            node: The node to move
            target_cycle: The desired cycle to move the node to
            schedule: The current schedule being modified
            resource_tracker: Resource tracker to validate resource constraints
            node_rrt_getter: Function to get resource requirements for nodes
            forward: If True, try cycles forward; if False, try cycles backward

        Returns:
            bool: True if the move was successful, False otherwise
        """
        original_cycle = schedule[node]
        resource_tracker.remove_node(node, original_cycle, node_rrt_getter)

        if forward:
            for try_cycle in range(target_cycle, target_cycle + self.T):
                if self._is_valid_move(
                    node, try_cycle, schedule, resource_tracker, node_rrt_getter
                ):
                    schedule[node] = try_cycle
                    resource_tracker.add_node(node, try_cycle, node_rrt_getter)
                    return True
        else:
            for try_cycle in range(target_cycle, target_cycle - self.T, -1):
                if self._is_valid_move(
                    node, try_cycle, schedule, resource_tracker, node_rrt_getter
                ):
                    schedule[node] = try_cycle
                    resource_tracker.add_node(node, try_cycle, node_rrt_getter)
                    return True

        schedule[node] = original_cycle
        resource_tracker.add_node(node, original_cycle, node_rrt_getter)
        return False

    def _is_valid_move(
        self,
        node: fx.Node,
        cycle: int,
        schedule: Schedule,
        resource_tracker: ResourceUsageTracker,
        node_rrt_getter: NodeRRTGetter,
    ) -> bool:
        if not resource_tracker.can_add_node(node, cycle, node_rrt_getter):
            return False

        for succ in self.graph.get_successors(node):
            if (
                succ in schedule
                and schedule[succ] <= cycle
                and self.graph.has_edge(node, succ)
            ):
                return False
        for pred in self.graph.get_predecessors(node):
            if (
                pred in schedule
                and schedule[pred] >= cycle
                and self.graph.has_edge(pred, node)
            ):
                return False
        return True

    def repair_forward(
        self,
        schedule: Schedule,
        resource_tracker: ResourceUsageTracker,
        node_rrt_getter: NodeRRTGetter,
    ) -> Tuple[bool, Schedule]:
        return self._repair_schedule(
            schedule, resource_tracker, node_rrt_getter, forward=True
        )

    def repair_backward(
        self,
        schedule: Schedule,
        resource_tracker: ResourceUsageTracker,
        node_rrt_getter: NodeRRTGetter,
    ) -> Tuple[bool, Schedule]:
        return self._repair_schedule(
            schedule, resource_tracker, node_rrt_getter, forward=False
        )

    def validate_dependencies(self, schedule: Schedule) -> bool:
        return all(
            schedule[edge._to] > schedule[edge._from]
            for edge in self.edges
            if edge._from in schedule and edge._to in schedule
        )


class ScheduleValidator:
    """Validates and optimizes operation schedules while maintaining resource and dependency constraints."""

    def __init__(
        self,
        initial_schedule: Schedule,
        T: int,
        nodes: List[fx.Node],
        resource_limits: np.ndarray,
        node_rrt_getter: NodeRRTGetter,
        raw_edges_list: List[Edge],
        num_resource_types: int,
    ):
        self.nodes = nodes
        self.T = T
        self.resource_limits = resource_limits
        self.node_rrt_getter = node_rrt_getter
        self.num_resource_types = num_resource_types
        self.edges = raw_edges_list
        self.S = initial_schedule.copy()
        self._dep_graph = ScheduleDependencyGraph(self.nodes, self.edges)
        self._resource_tracker = ResourceUsageTracker(
            self.resource_limits, self.T, self.num_resource_types
        )

        # Initialize resource tracker with initial schedule
        for node, start_cycle in self.S.items():
            self._resource_tracker.add_node(node, start_cycle, self.node_rrt_getter)

        self._repairer = ScheduleConstraintRepairer(self._dep_graph, self.edges, self.T)

    def attempt_move(
        self, node_to_move: fx.Node, new_cycle: int
    ) -> Tuple[bool, Optional[Schedule], Optional[str]]:
        if node_to_move not in self.S:
            return False, None, "Node not in schedule"

        if new_cycle == self.S[node_to_move]:
            return True, self.S.copy(), None

        repaired_schedule = self.S.copy()
        resource_tracker_candidate = ResourceUsageTracker(
            self.resource_limits, self.T, self.num_resource_types
        )

        # Initialize resource tracker with all nodes except the one being moved
        for node, start_cycle in repaired_schedule.items():
            if node != node_to_move:
                resource_tracker_candidate.add_node(
                    node, start_cycle, self.node_rrt_getter
                )

        repaired_schedule[node_to_move] = new_cycle
        success, repaired_schedule = self._repairer._repair_schedule(
            repaired_schedule,
            resource_tracker_candidate,
            self.node_rrt_getter,
            forward=(new_cycle > self.S[node_to_move]),
        )

        if not success:
            return False, None, "Schedule repair failed"

        if not self._repairer.validate_dependencies(repaired_schedule):
            return False, None, "Dependency validation failed after repair"

        return True, repaired_schedule, None

    def commit_move(self, new_schedule: Schedule, new_rt: np.ndarray) -> None:
        """Commits a previously validated schedule change to make it permanent.

        This method is called after a successful attempt_move to finalize a schedule change.
        It updates both the schedule and resource tracking state to the new validated state.

        The method implements a try-commit pattern where:
        1. attempt_move tries out and validates a schedule change
        2. If successful, commit_move makes that change permanent

        Args:
            new_schedule: The new schedule that was successfully validated by attempt_move
            new_rt: The corresponding resource tracking state for the new schedule

        Raises:
            ValueError: If the new schedule is invalid (wrong number of nodes or invalid nodes)
        """
        if len(new_schedule) != len(self.S) or not all(
            n in self.nodes for n in new_schedule.keys()
        ):
            raise ValueError("Invalid schedule for commit.")
        self.S = new_schedule
        self._resource_tracker.RT_global = np.array(new_rt)

    def get_current_schedule_state(self) -> Tuple[Schedule, np.ndarray]:
        return self.S, np.array(self._resource_tracker.RT_global)
