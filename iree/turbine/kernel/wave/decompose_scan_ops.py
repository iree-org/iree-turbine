from operator import ge

from sympy import Integer, Mod
import sympy
from iree.turbine.kernel._support.indexing import IndexExpr
from iree.turbine.kernel.compiler.builder import IRProxyValue
from iree.turbine.kernel.compiler.vector_codegen import cast_py_value
from iree.turbine.kernel.wave.codegen.emitter import add_emitter_subs, gen_sympy_index
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Broadcast,
    Conditional,
    SelectOp,
    get_custom,
    CustomOp,
    ScanOp,
    Cumsum,
    Add,
    ShuffleOp,
    Extract,
    select,
    log2,
    CastOp,
    register,
    NewRegister,
)
from .._support.dtype import DataType, i1
from ..lang.global_symbols import *
from ..wave.constraints import HardwareConstraint
from .utils.graph_utils import DCE
from typing import Callable
import torch.fx as fx
import math
from ..compiler.ir import gpu_d, arith_d, F32Type, vector_d
from ..compiler.ir import VectorType
from ..lang.wave_types import Memory, Register


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    return custom.fx_node


def insert_conditional_node(cmp: Conditional, graph: fx.Graph) -> fx.Node:
    cmp.graph = graph
    cmp.fx_node = graph.create_node(
        "call_function",
        target=cmp._tracing_function,
        args=(
            cmp.condition,
            cmp.subgraph_name,
            cmp.implicit_captures,
            cmp.if_true,
            cmp.if_false,
        ),
        kwargs={},
    )
    cmp.fx_node.tkw_op = cmp.__class__
    cmp.fx_node.tkw_op_name = cmp.tkw_op_name
    cmp.fx_node.type = get_custom(cmp.if_true).type
    return cmp.fx_node


def insert_select_node(
    cond: fx.Node, if_true: fx.Node, if_false: fx.Node, graph: fx.Graph
) -> fx.Node:
    sel = SelectOp(cond, if_true, if_false)
    sel.graph = graph
    sel.fx_node = graph.create_node(
        "call_function",
        target=sel._tracing_function,
        args=(cond, if_true, if_false),
        kwargs={},
    )
    sel.fx_node.tkw_op = sel.__class__
    sel.fx_node.tkw_op_name = sel.tkw_op_name
    sel.fx_node.type = get_custom(if_true).type
    return sel.fx_node


def create_symbolic_condition_select(cond_expr, if_true, if_false, shape, graph):
    def cond_lambda():
        return cond_expr

    cond_node = graph.call_function(cond_lambda, args=())
    cond_reg = register(shape, i1, cond_node)
    cond_node_fx = cond_reg.add_to_graph(graph)

    select_op = SelectOp(cond=cond_node_fx, if_true=if_true, if_false=if_false)
    return select_op.add_to_graph(graph)


def zero_like(node: fx.Node, graph: fx.Graph) -> fx.Node:
    shape = get_custom(node).type.symbolic_shape
    dtype = get_custom(node).type.dtype
    return get_graph_node(NewRegister(shape, dtype, 0.0), graph)


def emit_global_scan(
    binary_fn: Callable,
    src: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
    hardware_constraint: HardwareConstraint,
) -> fx.Node:
    """
    Emit an intra-warp inclusive scan using a butterfly pattern
    """
    init = src
    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx
        shuffle = ShuffleOp(init, offset_val, subgroup_size)
        shuffle_val = get_graph_node(shuffle, graph)
        lane_id = hardware_constraint.linearized_thread_id % subgroup_size
        zero_vec = zero_like(shuffle_val, graph)
        zero_vec.index = get_custom(src).index

        # zero_vec = get_graph_node(NewRegister(get_custom(init).type.symbolic_shape, get_custom(init).type.dtype, 0.0), graph)

        # cmp_i1 = Conditional(
        #     condition=ge(lane_id, Integer(offset_val)),
        #     subgraph_name=None,
        #     implicit_captures=(),
        #     if_true=shuffle_val,
        #     if_false=zero_vec,
        # )
        # cmp_i1 = insert_conditional_node(cmp_i1, graph)

        cond_expr = ge(lane_id, Integer(offset_val))
        cond_node = get_graph_node(
            NewRegister(get_custom(init).type.symbolic_shape, i1, cond_expr), graph
        )
        cond_node.index = get_custom(src).index

        masked = get_graph_node(
            SelectOp(cond=cond_node, if_true=shuffle_val, if_false=zero_vec), graph
        )

        init = get_graph_node(binary_fn(init, masked), graph)

    return init


def decompose_scan_ops(
    trace: CapturedTrace,
    constraints: list,
):
    scan_nodes = trace.walk(lambda node: isinstance(get_custom(node), ScanOp))
    if not scan_nodes:
        return

    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    subgroup_size = hardware_constraint.threads_per_wave

    for node in scan_nodes:
        custom = get_custom(node)
        if not isinstance(custom, Cumsum):
            raise NotImplementedError(f"ScanOp '{custom}' not supported")

        with custom.graph.inserting_before(custom.fx_node):
            src, acc, dim = node.args
            assert isinstance(src, fx.Node), f"Scan src is not fx.Node: {type(src)}"
            binary_fn = Add

            result = emit_global_scan(
                binary_fn, src, custom.graph, subgroup_size, hardware_constraint
            )
            custom.replace_all_uses_with(result)

    DCE(trace)
