from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from .address_spaces import *
import shark_turbine.kernel.lang as tkl

logger = get_logger("turbine.wave.hoisting")


def get_allocs_(graph: fx.Graph) -> list[fx.Node]:
    allocs = []
    for node in graph.nodes:
        if hasattr(node, "tkw_op") and node.tkw_op == Allocate:
            allocs.append(node)
    return allocs


def hoist_allocs(trace: CapturedTrace):
    """Hoists allocs from reduction subgraphs to outer root graph."""
    root_graph = trace.get_root_graph()
    for node in root_graph.nodes:
        custom_node = get_custom(node)
        match custom_node:
            case Reduction():
                with root_graph.inserting_before(custom_node.fx_node):
                    subgraph = trace.get_subgraph(custom_node.subgraph_name)
                    allocs = get_allocs_(subgraph)
                    for alloc in allocs:
                        new_alloc = root_graph.node_copy(alloc)
                        alloc.replace_all_uses_with(new_alloc)
                        subgraph.erase_node(alloc)
