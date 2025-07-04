from .._support.tracing import CapturedTrace
from .._support.indexing import *
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .constraints import *
from .utils.symbol_utils import *


def reorder_workgroups(graph: CapturedTrace, reordering_constraints):
    new_wg0, new_wg1, new_wg2 = None, None, None
    for c in reordering_constraints:
        if c.wg_dim == WORKGROUP_0:
            new_wg0 = c.reordered_equation
        elif c.wg_dim == WORKGROUP_1:
            new_wg1 = c.reordered_equation
        else:
            new_wg2 = c.reordered_equation

    if new_wg0 is not None or new_wg1 is not None or new_wg2 is not None:
        graph_nodes = graph.walk()
        for node in graph_nodes:
            custom_node = get_custom(node)
            if custom_node.index:
                op_set = {"iterate", "get_result"}
                if custom_node.name not in op_set:
                    wg0, wg1, wg2 = WORKGROUP_0, WORKGROUP_1, WORKGROUP_2
                    for dim, symb_exp in node.index.items():
                        symbols = symb_exp.start.free_symbols
                        if wg0 in symbols and new_wg0 is not None:
                            symb_exp.start = safe_subs(symb_exp.start, {wg0: new_wg0})
                        elif wg1 in symbols and new_wg1 is not None:
                            symb_exp.start = safe_subs(symb_exp.start, {wg1: new_wg1})
                        elif wg2 in symbols and new_wg2 is not None:
                            symb_exp.start = safe_subs(symb_exp.start, {wg2: new_wg2})
