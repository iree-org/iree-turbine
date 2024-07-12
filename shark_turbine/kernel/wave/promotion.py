from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from .address_spaces import *
import shark_turbine.kernel.lang as tkl

logger = get_logger("turbine.wave.promotion")


def apply_promotion_pattern_(
    custom_node: Read | Write, allocate_node: Allocate, graph: fx.Graph
) -> list[fx.Node]:
    promoted_nodes = []
    match custom_node:
        case Read(
            memory, elements_per_thread
        ) if memory.type.address_space != allocate_node.address_space:
            promoted_write = Write(
                custom_node.fx_node, allocate_node.fx_node, elements_per_thread
            ).add_to_graph(graph)
            promoted_read = Read(
                allocate_node.fx_node, elements_per_thread
            ).add_to_graph(graph)
            promoted_nodes = [promoted_write, promoted_read]
            custom_node.fx_node.replace_all_uses_with(promoted_read)
    return promoted_nodes


def promote_node(node: fx.Node, graph: fx.Graph, address_space: IndexSymbol):
    """Promotes the given operand in the provided graph
    to the specified address space.

    The process of promotion involves allocating memory
    in the new address space and writing to the new
    memory location and subsequent uses reading from there.
    """

    custom_node = get_custom(node)
    assert isinstance(custom_node, Read) or isinstance(custom_node, Write)
    with graph.inserting_before(node.next):
        allocate_node = Allocate(
            custom_node.type.symbolic_shape, custom_node.type.dtype, address_space
        )
        allocate_node.add_to_graph(graph)
        apply_promotion_pattern_(custom_node, allocate_node, graph)
