from .._support.tracing import CapturedTrace
from ...support.logging import get_logger
from ..ops.wave_ops import *

logger = get_logger("turbine.wave.register_analysis")


def set_register_shape(trace: CapturedTrace, custom: CustomOp) -> None:
    for custom_user in custom.users:
        if isinstance(custom_user, MMA):
            arg_index = custom_user.fx_node.args.index(custom.fx_node)
            match arg_index:
                case 0:
                    custom.fx_node.thread_shape = custom_user.lhs_index[0].size
                case 1:
                    custom.fx_node.thread_shape = custom_user.rhs_index[0].size
                case 2:
                    custom.fx_node.thread_shape = custom_user.acc_index[0].size
            break

        elif isinstance(custom_user, Reduction):
            idx = custom_user.init_args.index(custom.fx_node)
            iter_arg = get_custom(
                custom_user.iter_args(trace.get_subgraph(custom_user.subgraph_name))[
                    idx
                ]
            )
            set_register_shape(trace, iter_arg)
            custom.fx_node.thread_shape = iter_arg.fx_node.thread_shape
            break
        else:
            raise NotImplementedError(
                f"Register shape propagation not implemented for {custom_user}"
            )


def determine_register_shape(trace: CapturedTrace | fx.Graph) -> None:
    """
    Each register op is annotated with the wave shape of the register. This
    function determines the thread shape of the register based on the uses
    of the register in the graph.
    """
    register_nodes = trace.walk(lambda node: isinstance(get_custom(node), NewRegister))
    if not register_nodes:
        return
    for node in register_nodes:
        set_register_shape(trace, get_custom(node))
