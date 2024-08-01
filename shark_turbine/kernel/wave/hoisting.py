from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.hoisting")


def get_allocs(graph: fx.Graph) -> list[CustomOp]:
    return [
        custom_node
        for node in graph.nodes
        if isinstance((custom_node := get_custom(node)), Allocate)
    ]


def hoist_allocs(trace: CapturedTrace):
    """Hoists allocs from reduction subgraphs to outer root graph."""
    root_graph = trace.get_root_graph()
    for node in root_graph.nodes:
        custom_node = get_custom(node)
        match custom_node:
            case Reduction():
                with root_graph.inserting_before(custom_node.fx_node):
                    subgraph = trace.get_subgraph(custom_node.subgraph_name)
                    allocs = get_allocs(subgraph)
                    for alloc in allocs:
                        new_alloc = alloc.copy(new_graph=root_graph)
                        alloc.replace_all_uses_with(new_alloc)
                        alloc.erase()
