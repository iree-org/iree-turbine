import math
from operator import ge
from typing import Any, Callable

import torch.fx as fx

from iree.turbine.kernel._support.indexing import IndexSymbol
from iree.turbine.kernel.wave.utils.symbol_utils import subs_idxc

from .._support.dtype import i1
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Add,
    Cumsum,
    CustomOp,
    Extract,
    NewRegister,
    Reshape,
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


def emit_local_inclusive_scan(
    binary_fn: Callable,
    scan_src: fx.Node,
    graph: fx.Graph,
    elements_per_thread: int,
) -> list[fx.Node]:
    """
    Perform local inclusive scan for `n` elements per thread.
    """
    values = [
        get_graph_node(Extract(scan_src, [i]), graph)
        for i in range(elements_per_thread)
    ]

    for i in range(1, elements_per_thread):
        values[i] = get_graph_node(binary_fn(values[i], values[i - 1]), graph)

    return values


def emit_global_scan_multiple_elements(
    binary_fn: Callable,  # Supports only Add for now.
    src: fx.Node,
    local_scan: list[fx.Node],
    graph: fx.Graph,
    subgroup_size: int,
    hardware_constraint: HardwareConstraint,
    local_scan_size: int,
    scan_dim: IndexSymbol,
) -> fx.Node:
    """
    Emit an intra-warp inclusive scan using butterfly pattern scan and masking.
    """
    offset = local_scan[-1]
    lane_id = (
        hardware_constraint.linearized_thread_id % hardware_constraint.threads_per_wave
    )

    target_shape = list(src.type.symbolic_shape)
    target_shape.pop(target_shape.index(scan_dim))

    thread_incl = offset
    thread_incl.index = {target_shape[0]: get_custom(src).index[target_shape[0]]}

    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx

        # shuffle operation to get value from another thread
        shuffle = ShuffleOp(thread_incl, offset_val, subgroup_size, ShuffleMode.UP)
        shuffle_val_node = get_graph_node(shuffle, graph)

        # we are explicitly adding index because this pass is being
        # applied after the indexing phase
        zero_vec = get_graph_node(
            NewRegister(
                get_custom(shuffle_val_node).type.symbolic_shape,
                get_custom(shuffle_val_node).type.dtype,
                0.0,
            ),
            graph,
        )
        zero_vec.index = get_custom(local_scan[-1]).index

        # condition node: thread ID >= offset
        cond_expr = ge(lane_id, offset_val)
        cond_node = get_graph_node(
            NewRegister(get_custom(offset).type.symbolic_shape, i1, cond_expr), graph
        )
        cond_node.index = get_custom(local_scan[-1]).index

        # apply shuffle_val only if condition is true; else use 0
        masked = get_graph_node(
            SelectOp(cond=cond_node, if_true=shuffle_val_node, if_false=zero_vec), graph
        )

        thread_incl = get_graph_node(binary_fn(thread_incl, masked), graph)

    reshape = thread_incl

    # Phase 2 to perform global scan
    sh1 = ShuffleOp(reshape, 1, subgroup_size, ShuffleMode.UP)
    sh1_n = get_graph_node(sh1, graph)

    cond1 = ge(lane_id, 1)
    cond1_n = get_graph_node(
        NewRegister(get_custom(thread_incl).type.symbolic_shape, i1, cond1), graph
    )
    cond1_n.index = get_custom(local_scan[-1]).index

    zero1 = get_graph_node(
        NewRegister(
            get_custom(sh1_n).type.symbolic_shape, get_custom(sh1_n).type.dtype, 0.0
        ),
        graph,
    )
    zero1.index = get_custom(local_scan[-1]).index

    excl_offset = get_graph_node(
        SelectOp(cond=cond1_n, if_true=sh1_n, if_false=zero1), graph
    )

    final_scalars = []
    for lane_elem in local_scan:
        summed = get_graph_node(binary_fn(lane_elem, excl_offset), graph)
        summed.index = get_custom(src).index
        final_scalars.append(summed)

    # pack the output in the expected order
    reshape = Reshape(
        args=final_scalars,
        target_vector_shape={scan_dim: local_scan_size},
    ).add_to_graph(graph)

    reshape.index = get_custom(src).index
    reshape.expanded_dims = get_custom(src).expanded_dims
    reshape.vector_shapes = get_custom(src).vector_shapes

    return reshape


def emit_global_scan_per_element(
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
            scan_src, scan_acc, scan_dim = node.args
            assert isinstance(
                scan_src, fx.Node
            ), f"Scan src is not fx.Node: {type(scan_src)}"
            binary_fn = Add

            if scan_dim is None:
                raise ValueError("No scan dim specified, please specify a scan dim.")

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )

            try:
                op = get_custom(scan_src)
                thread_shape = get_thread_shape(op.index)
                local_scan_sizes = thread_shape
            except Exception as e:
                index_str = "\n".join(f"{k}: {v}" for k, v in op.index.items())
                raise RuntimeError(
                    f"Error in decompose_scan_ops: {scan_src} with index\n"
                    f"{index_str}\n{scan_src=}\n{scan_acc=}\n{scan_dim=}"
                ) from e

            if local_scan_sizes > 1:
                local_scan = emit_local_inclusive_scan(
                    binary_fn, scan_src, custom.graph, local_scan_sizes
                )
                result = emit_global_scan_multiple_elements(
                    binary_fn,
                    scan_src,
                    local_scan,
                    custom.graph,
                    subgroup_size,
                    hardware_constraint,
                    local_scan_sizes,
                    scan_dim,
                )
            else:
                result = emit_global_scan_per_element(
                    binary_fn,
                    scan_src,
                    custom.graph,
                    subgroup_size,
                    hardware_constraint,
                )

            custom.replace_all_uses_with(result)

    DCE(trace)
