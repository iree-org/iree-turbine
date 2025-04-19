from operator import ge
from iree.turbine.kernel.compiler.vector_codegen import cast_py_value
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Broadcast,
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
from ..compiler.ir import gpu_d, arith_d, F32Type, IndexType, vector_d
from ..compiler.ir import VectorType
from ..lang.wave_types import Memory, Register


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    return custom.fx_node


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
        shuffle_val = get_graph_node(ShuffleOp(init, offset_val, subgroup_size), graph)
        lane_id = hardware_constraint.linearized_thread_id % subgroup_size
        zero_vec = get_graph_node(
            NewRegister(
                get_custom(init).type.symbolic_shape, get_custom(init).type.dtype, 0.0
            ),
            graph,
        )

        cmp = graph.call_function(
            lambda: arith_d.cmpi(arith_d.CmpIPredicate.sge, lane_id, offset_val),
            args=(),
        )
        cmp_i1 = CastOp(cmp, i1)
        cmp_i1 = get_graph_node(
            NewRegister(get_custom(init).type.symbolic_shape, i1, CastOp(cmp, i1)),
            graph,
        )

        # cmp_i1.type = Register[(*get_custom(init).type.symbolic_shape, i1)]

        # cmp_i1 = get_graph_node(cmp_i1, graph)
        # cmp = get_graph_node(ge(lane_id, offset_val), graph)

        masked = get_graph_node(SelectOp(cmp_i1, shuffle_val, zero_vec), graph)
        init = get_graph_node(binary_fn(init, masked), graph)

        # init = get_graph_node(binary_fn(init, shuffle_val), graph)
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
            breakpoint()
            custom.replace_all_uses_with(result)

    DCE(trace)
