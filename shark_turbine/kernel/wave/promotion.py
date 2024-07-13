from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from .address_spaces import *
import shark_turbine.kernel.lang as tkl

logger = get_logger("turbine.wave.promotion")


def apply_promotion_pattern_(custom_node: Read | Write, allocate_node: Allocate):
    match custom_node:
        case Read(
            memory, elements_per_thread
        ) if memory.type.address_space != allocate_node.address_space:
            promoted_read = Read(
                allocate_node.fx_node, elements_per_thread
            ).add_to_graph(custom_node.graph)
            custom_node.replace_all_uses_with(promoted_read)
            with custom_node.graph.inserting_before(promoted_read):
                Write(
                    custom_node.fx_node, allocate_node.fx_node, elements_per_thread
                ).add_to_graph(custom_node.graph)


def promote_node(node: CustomOp, address_space: IndexSymbol):
    """Promotes the given operand in the provided graph
    to the specified address space.

    The process of promotion involves allocating memory
    in the new address space and writing to the new
    memory location and subsequent uses reading from there.
    """

    assert isinstance(node, Read) or isinstance(node, Write)
    with node.graph.inserting_before(node.fx_node.next):
        allocate_node = Allocate(
            node.type.symbolic_shape, node.type.dtype, address_space
        )
        allocate_node.add_to_graph(node.graph)
        apply_promotion_pattern_(node, allocate_node)
