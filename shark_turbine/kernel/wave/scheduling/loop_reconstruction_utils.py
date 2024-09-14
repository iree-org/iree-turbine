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

    def __init__(self, graph: fx.Graph, num_stages: int) -> None:
        self.argument_map: list[dict[fx.Node, fx.Node]] = [
            {} for _ in range(num_stages)
        ]
        self.graph = graph
        self.output_node = graph._root.prev
        assert isinstance(get_custom(self.output_node), Output)
        self.results = get_custom(self.output_node).return_vals[0]

    def map_arg(self, stage: int, from_: fx.Node, to_: fx.Node) -> None:
        """
        Maps the given argument from one node to another in the argument context
        at the given stage.
        """
        assert stage < len(self.argument_map), f"Stage {stage} not yet initialized"
        self.argument_map[stage][from_] = to_

    def map_arg_all_stages(self, from_: fx.Node, to_: fx.Node) -> None:
        """
        Maps the given argument from one to another into the argument context for all stages.
        """
        for stage in range(len(self.argument_map)):
            self.argument_map[stage][from_] = to_

    def query_arg(self, stage: int, from_: fx.Node) -> Optional[fx.Node]:
        """
        Queries the argument mapping for the given stage and node.
        """
        if stage >= len(self.argument_map):
            return None
        return self.argument_map[stage].get(from_, None)

    def query_mapped_results(self) -> list[fx.Node]:
        """
        Queries the mapped results for all stages.
        """
        mapped_results = []
        for result in self.results:
            for mapping in self.argument_map:
                if result in mapping:
                    mapped_results.append(mapping[result])
                    break
        return mapped_results

    def __contains__(self, key: fx.Node | tuple[int, fx.Node]) -> bool:
        """
        Checks if the argument context contains the given node at a specified
        stage or at all stages.
        """
        if isinstance(key, tuple):
            stage, key = key
            return key in self.argument_map[stage]
        return any(key in stage for stage in self.argument_map)
