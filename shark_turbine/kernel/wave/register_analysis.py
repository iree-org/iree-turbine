from .._support.tracing import CapturedTrace
from ...support.logging import get_logger
from ..ops.wave_ops import *

logger = get_logger("turbine.wave.register_analysis")


def determine_register_shape(trace: CapturedTrace) -> None:
    """
    Each register op is annotated with the wave shape of the register. This
    function determines the thread shape of the register based on the uses
    of the register in the graph.
    """
    register_nodes = trace.walk(lambda node: isinstance(get_custom(node), NewRegister))
    if not register_nodes:
        return
    for node in register_nodes:
        custom_node = get_custom(node)
        for user in node.users.keys():
            custom_user = get_custom(user)
            if isinstance(custom_user, MMA):
                arg_index = user.args.index(node)
                if arg_index == 0:
                    custom_node.fx_node.thread_shape = custom_user.lhs_index[0].size
                if arg_index == 1:
                    custom_node.fx_node.thread_shape = custom_user.rhs_index[0].size
                if arg_index == 2:
                    custom_node.fx_node.thread_shape = custom_user.acc_index[0].size
            else:
                raise NotImplementedError(
                    f"Register shape propagation not implemented for {user}"
                )
