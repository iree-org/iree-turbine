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
)
from ..lang.global_symbols import *
from ..wave.constraints import HardwareConstraint
from .utils.graph_utils import DCE
from typing import Callable
import torch.fx as fx
import math
from ..compiler.ir import gpu_d, arith_d, F32Type, IndexType, vector_d
from ..compiler.ir import VectorType


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    return custom.fx_node


def emit_global_scan(
    binary_fn: Callable,
    src: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
) -> fx.Node:
    """
    Emit an intra-warp inclusive scan using a butterfly pattern
    """
    init = src
    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx
        shuffle_val = get_graph_node(ShuffleOp(init, offset_val, subgroup_size), graph)

        lane_id = graph.call_function(
            lambda: gpu_d.thread_id(gpu_d.Dimension.x), args=()
        )
        offset_const = graph.call_function(
            lambda: cast_py_value(offset_val, IndexType.get()), args=()
        )

        cmp = graph.call_function(
            lambda: arith_d.cmpi(arith_d.CmpIPredicate.sge, lane_id, offset_const),
            args=(),
        )
        zero_vec = graph.call_function(
            lambda: cast_py_value(0.0, VectorType.get([subgroup_size], F32Type.get())),
            args=(),
        )

        masked = graph.call_function(
            lambda: arith_d.select(cmp, shuffle_val, zero_vec), args=()
        )
        init = get_graph_node(
            binary_fn(init, masked), graph
        )  ## Error: AttributeError: 'NoneType' object has no attribute 'symbolic_shape'

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

            result = emit_global_scan(binary_fn, src, custom.graph, subgroup_size)
            custom.replace_all_uses_with(result)

    DCE(trace)
