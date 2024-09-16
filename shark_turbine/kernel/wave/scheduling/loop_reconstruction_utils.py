from ..constraints import Constraint, TilingConstraint
from ..._support.indexing import IndexSymbol
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, Output, Write, GetResult, get_custom
from .modulo_scheduling import ModuloScheduler
from ..utils import graph_copy, erase_graph
from ..utils import subs_idxc
import torch.fx as fx
import math
from collections import defaultdict, deque, ChainMap
from ..visualization import visualize_mapped_graphs
from ....support.logging import get_logger
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
import random
from typing import Optional

logger = get_logger("turbine.wave.scheduling.loop_reconstruction_utils")


class ArgumentContext:
    """
    The argument context is used to store the mapping of arguments
    for each modulo pipelining stage.
    """

    def __init__(
        self, results: list[fx.Node], iter_args: list[fx.Node], num_stages: int
    ) -> None:
        self.argument_map: list[dict[fx.Node, fx.Node]] = [
            {} for _ in range(num_stages)
        ]
        self.results = results
        self.iter_args = iter_args
        self.result_to_iter_arg: dict[fx.Node, fx.Node] = {}
        for result, iter_arg in zip(results, iter_args):
            self.result_to_iter_arg[result] = iter_arg

    def map_arg_all_stages(self, from_: fx.Node, to_: fx.Node) -> None:
        """
        Maps the given argument from one to another into the argument context for all stages.
        """
        for stage in range(len(self.argument_map)):
            self.argument_map[stage][from_] = to_

    def get_mapped_results(self) -> list[fx.Node]:
        """
        Gets the mapped results for all stages.
        """
        mapped_results = []
        for result in self.results:
            for mapping in self.argument_map:
                if result in mapping:
                    mapped_results.append(mapping[result])
                    break
        return mapped_results

    def __setitem__(self, key: tuple[int, fx.Node], value: fx.Node) -> None:
        """
        Sets the argument mapping for the given stage.
        """
        assert isinstance(key, tuple), "Argument context key must be a tuple"
        stage, from_ = key
        assert stage < len(self.argument_map), f"Stage {stage} not yet initialized"
        self.argument_map[stage][from_] = value

    def __getitem__(self, value: tuple[int, fx.Node]) -> fx.Node:
        """
        Gets the argument mapping for the given stage.
        """
        assert isinstance(value, tuple), "Argument context key must be a tuple"
        stage, key = value
        assert stage < len(self.argument_map), f"Stage {stage} not yet initialized"
        return self.argument_map[stage].get(key, None)

    def __contains__(self, key: fx.Node | tuple[int, fx.Node]) -> bool:
        """
        Checks if the argument context contains the given node at a specified
        stage or at all stages.
        """
        if isinstance(key, tuple):
            stage, key = key
            return key in self.argument_map[stage]
        return any(key in stage for stage in self.argument_map)


def create_fill_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the prologue (fill).
    This looks like:
    [0 None None None]
    [1    0 None None]
    [2    1    0 None]
    """
    schedule = []
    for i in range(n - 1):
        row = list(range(i, -1, -1))
        row.extend([None] * (n - i - 1))
        schedule.append(row)
    return schedule


def create_drain_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the epilogue (drain).
    This looks like:
    [None    3    2 1]
    [None None    3 2]
    [None None None 3]
    """
    schedule = []
    for i in range(n - 1):
        row = [None] * (i + 1)
        row.extend(range(n - 1, i, -1))
        schedule.append(row)
    return schedule


def liveness_analysis(
    graph: fx.Graph, constraints: list[Constraint], scheduler: ModuloScheduler
) -> dict[fx.Node, int]:
    """
    Perform liveness analysis on the graph to determine the live ranges of
    variables and use that to deduce how many rotating registers we need.
    """
    lifetime: dict[fx.Node, int] = {}
    for node in graph.nodes:
        custom = get_custom(node)
        if custom.scheduling_parameters is None:
            continue
        if node not in lifetime:
            lifetime[node] = 0
        for user in custom.users:
            if user.scheduling_parameters is None:
                continue
            logger.debug(
                f"Node: {node}, User: {user.fx_node}, lifetime: {user.scheduling_parameters['stage'] - custom.scheduling_parameters['stage']}"
            )
            lifetime[node] = max(
                user.scheduling_parameters["stage"]
                - custom.scheduling_parameters["stage"],
                lifetime[node],
            )

    # Determine how many copies we need for each node. If the lifetime of a node
    # is l clocks and the initiation interval is T, then only ceil(l/T) values
    # of the node can be live at the same time. We need to create copies of only
    # those nodes that are live at more than one stage.
    num_rotating_registers: dict[fx.Node, int] = {}
    for node, l in lifetime.items():
        if node in num_rotating_registers:
            continue
        custom = get_custom(node)
        if (
            isinstance(custom, Write)
            and custom.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            continue
        if l > 0:
            num_rotating_registers[node] = l

    return num_rotating_registers


def partition_graph_by_stage(
    graph: fx.Graph, scheduler: ModuloScheduler
) -> list[dict[int, list[fx.Node]]]:
    """
    Partition the graph into stages based on the scheduling parameters.
    """
    partitioned_graph: list[dict[int, list[fx.Node]]] = [
        defaultdict(list) for _ in range(scheduler.num_stages)
    ]
    for stage in range(scheduler.num_stages):
        for node in graph.nodes:
            custom = get_custom(node)
            if custom.scheduling_parameters is None:
                continue
            if isinstance(custom, IterArg):
                continue
            if custom.scheduling_parameters["stage"] == stage:
                cycle = custom.scheduling_parameters["cycle"]
                partitioned_graph[stage][cycle].append(node)
    return partitioned_graph


def interleave_instructions(instructions: list[tuple[int, int, fx.Node]]):
    """
    Interleave the instructions that are scheduled in the same cycle.
    Currently, we just randomly shuffle them, but we could also sort
    them based on some criteria.
    """
    rng = random.Random(0)
    rng.shuffle(instructions)
