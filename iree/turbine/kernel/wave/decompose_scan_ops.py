import math
from operator import ge
from typing import Any, Callable, Optional

import sympy
import torch.fx as fx

from .._support.dtype import i1

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Add,
    Cumsum,
    CustomOp,
    NewRegister,
    ScanOp,
    SelectOp,
    ShuffleOp,
    get_custom,
)
from .constraints import HardwareConstraint
from .utils.classes import ShuffleMode
from .utils.graph_utils import DCE


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    return custom.fx_node


def emit_global_scan(
    binary_fn: Callable,  # Supports only Add for now.
    src: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
    hardware_constraint: HardwareConstraint,
) -> fx.Node:
    """
    Emit an intra-warp inclusive scan using butterfly pattern scan and masking.
    """
    init = src
    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx

        # shuffle operation to get value from another thread
        shuffle = ShuffleOp(init, offset_val, subgroup_size, ShuffleMode.UP)
        shuffle_val = get_graph_node(shuffle, graph)

        lane_id = hardware_constraint.linearized_thread_id % subgroup_size

        # we are explicitly adding index because this pass is being
        # applied after the indexing phase
        # ToDo (xintin): check if we can replace register with scalar.
        # No point using register for a scalar. Applies to other objects too.
        zero_vec = get_graph_node(
            NewRegister(
                get_custom(shuffle_val).type.symbolic_shape,
                get_custom(shuffle_val).type.dtype,
                0.0,
            ),
            graph,
        )

        # We are explicitly setting the indices to avoid:
        # AttributeError: 'NoneType' object has no attribute 'values'
        # ToDo (xintin): After testing with cherry-picks locally, I can say that we will
        # not need these explicit setter after block reduce PRs. I will revisit.
        zero_vec.index = get_custom(src).index

        # condition node: thread ID >= offset
        cond_expr = ge(lane_id, offset_val)
        cond_node = get_graph_node(
            NewRegister(get_custom(init).type.symbolic_shape, i1, cond_expr), graph
        )
        cond_node.index = get_custom(src).index

        # apply shuffle_val only if condition is true; else use 0
        masked = get_graph_node(
            SelectOp(cond=cond_node, if_true=shuffle_val, if_false=zero_vec), graph
        )

        init = get_graph_node(binary_fn(init, masked), graph)

    return init


def decompose_scan_ops(
    trace: CapturedTrace,
    constraints: list,
) -> None:
    """Decomposes high-level scan operations (ScanOp) in a captured FX trace
    into lower-level warp-level inclusive scan implementations.

    Currently only supports Cumsum operations using a butterfly-style scan.

    Args:
        trace (CapturedTrace): fx trace representing the computation graph.
        constraints (list): list of hardware-related constraints

    Raises:
        NotImplementedError: If the scan operation type is not yet supported.
    """
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
            src, _, _ = node.args
            assert isinstance(src, fx.Node), f"Scan src is not fx.Node: {type(src)}"
            binary_fn = Add

            result = emit_global_scan(
                binary_fn, src, custom.graph, subgroup_size, hardware_constraint
            )

            custom.replace_all_uses_with(result)

    DCE(trace)
