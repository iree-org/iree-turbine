import math
from operator import ge
from typing import Callable

import torch.fx as fx

from iree.turbine.kernel._support.indexing import IndexSymbol
from iree.turbine.kernel.wave.utils.general_utils import all_equal
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
    scan_src: list[fx.Node],
    graph: fx.Graph,
    elements_per_thread: int,
) -> list[list[fx.Node]]:
    """
    Perform local inclusive scan for `n` elements per thread.
    """
    result = []
    for node in scan_src:
        values = [
            get_graph_node(Extract(node, [i]), graph)
            for i in range(elements_per_thread)
        ]
        values[0].index = node.index

        for i in range(1, elements_per_thread):
            values[i] = get_graph_node(binary_fn(values[i], values[i - 1]), graph)
            values[i].index = node.index

        result.append(values)
    return result


def emit_variable_scan(
    binary_fn: Callable,
    src: list[list[fx.Node]],
    graph: fx.Graph,
    elements_per_thread: int,
) -> list[list[fx.Node]]:
    result: list[list[fx.Node]] = []
    result.append(src[0])

    for scan_src_idx in range(1, len(src)):
        prev_thread_last_val = result[scan_src_idx - 1][-1]
        temp: list[fx.Node] = []
        for i in range(elements_per_thread):
            cur_val = src[scan_src_idx][i]
            offset_val = get_graph_node(binary_fn(prev_thread_last_val, cur_val), graph)
            offset_val.index = get_custom(src[scan_src_idx][i]).index
            temp.append(offset_val)
        result.append(temp)

    return result


def emit_global_scan(
    binary_fn: Callable,  # Supports only Add for now.
    src: fx.Node,
    local_scan: list[list[fx.Node]],
    graph: fx.Graph,
    subgroup_size: int,
    hardware_constraint: HardwareConstraint,
    local_scan_size: int,
    scan_dim: IndexSymbol,
) -> list[fx.Node]:
    """
    Emit an intra-warp inclusive scan using butterfly pattern scan and masking.
    In this algorithm, we are using two-pass approach.
    If the number of elements per thread is one, then we just need one shuffle pass.
    If there are more than one elements per thread, we need to calculate the exclusive
        offset for the next lane elements using the information from the previous lane.
        That is why we have another shuffle phase.

    TODO: Can we make it more efficient?
    """
    lane_id = (
        hardware_constraint.linearized_thread_id % hardware_constraint.threads_per_wave
    )

    scanop_result = local_scan[-1][-1]
    last_local_scan_node = get_custom(scanop_result)

    target_shape = list(src.type.symbolic_shape)
    target_shape.pop(target_shape.index(scan_dim))
    scanop_result.index = {target_shape[0]: get_custom(src).index[target_shape[0]]}

    num_steps = int(math.log2(float(subgroup_size)))
    for idx in range(num_steps):
        offset_val = 1 << idx

        # shuffle operation to get value from another thread
        shuffle = ShuffleOp(scanop_result, offset_val, subgroup_size, ShuffleMode.UP)
        shuffle_val_node = get_graph_node(shuffle, graph)

        # we are explicitly adding index because this pass is being
        # applied after the indexing phase
        # TODO (xintin): check if we can replace register with scalar.
        # No point using register for a scalar. Applies to other objects too.
        zero_vec = get_graph_node(
            NewRegister(
                get_custom(shuffle_val_node).type.symbolic_shape,
                get_custom(shuffle_val_node).type.dtype,
                0.0,
            ),
            graph,
        )
        zero_vec.index = last_local_scan_node.index

        # condition node: thread ID >= offset
        cond_expr = ge(lane_id, offset_val)
        cond_node = get_graph_node(
            NewRegister(get_custom(scanop_result).type.symbolic_shape, i1, cond_expr),
            graph,
        )
        cond_node.index = last_local_scan_node.index

        # apply shuffle_val only if condition is true; else use 0
        masked = get_graph_node(
            SelectOp(cond=cond_node, if_true=shuffle_val_node, if_false=zero_vec), graph
        )

        # perform binary scan op
        scanop_result = get_graph_node(binary_fn(scanop_result, masked), graph)
        final_scanop_result = [scanop_result]

    if local_scan_size > 1:
        # Phase 2 to calculate exclusive offset from previous lane id.
        scan_offset = ShuffleOp(scanop_result, 1, subgroup_size, ShuffleMode.UP)
        scan_offset_node = get_graph_node(scan_offset, graph)

        lane_id_ge_one = ge(lane_id, 1)
        lane_id_ge_one_node = get_graph_node(
            NewRegister(
                get_custom(scanop_result).type.symbolic_shape, i1, lane_id_ge_one
            ),
            graph,
        )
        # TODO: debug why last_local_scan_node.index is setting to None if assigned outside.
        lane_id_ge_one_node.index = last_local_scan_node.index

        identity_vec_node = get_graph_node(
            NewRegister(
                get_custom(scan_offset_node).type.symbolic_shape,
                get_custom(scan_offset_node).type.dtype,
                0.0,
            ),
            graph,
        )
        identity_vec_node.index = last_local_scan_node.index

        excl_offset = get_graph_node(
            SelectOp(
                cond=lane_id_ge_one_node,
                if_true=scan_offset_node,
                if_false=identity_vec_node,
            ),
            graph,
        )

        # Update the prefix sum for each locally scanned element based on the updated offset
        final_scanop_result = []
        for loc_scan in local_scan:
            custom = get_custom(loc_scan[0])

            final_scalars = [
                get_graph_node(binary_fn(lane_elem, excl_offset), graph)
                for lane_elem in loc_scan
            ]

            # pack the output in the expected order
            scanop_result = Reshape(
                args=final_scalars,
                target_vector_shape={scan_dim: local_scan_size},
            ).add_to_graph(graph)

            # assign attributes
            scanop_result.index = custom.index
            scanop_result.expanded_dims = custom.expanded_dims
            scanop_result.vector_shapes = custom.vector_shapes

            final_scanop_result.append(scanop_result)

    return final_scanop_result


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
            binary_fn = Add

            if scan_dim is None:
                raise ValueError("No scan dim specified, please specify a scan dim.")

            if not isinstance(scan_src, (list, tuple)):
                scan_src = [scan_src]

            # sanity checks and check for the fastest dim
            src_fastest_dims = [
                get_custom(arg).type.symbolic_shape[-1] for arg in scan_src
            ]
            if not all_equal(src_fastest_dims):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same fastest dim."
                )
            if scan_dim is not src_fastest_dims[0]:
                raise NotImplementedError(
                    f"Only implemented reduction on fastest dimension. Got {scan_dim} and {src_fastest_dims}."
                    f"\n{custom}"
                )

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )

            local_scan_sizes = []
            for arg in scan_src:
                try:
                    op = get_custom(arg)
                    thread_shape = get_thread_shape(op.index)
                    local_scan_sizes.append(thread_shape)
                except Exception as e:
                    index_str = "\n".join(f"{k}: {v}" for k, v in op.index.items())
                    raise RuntimeError(
                        f"Error in decompose_scan_ops: {scan_src} with index\n"
                        f"{index_str}\n{scan_src=}\n{scan_acc=}\n{scan_dim=}"
                    ) from e

            if not all_equal(local_scan_sizes):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same local reduce size."
                )

            # Phase 1: Local scan
            local_scan = emit_local_inclusive_scan(
                binary_fn, scan_src, custom.graph, local_scan_sizes[0]
            )

            # Phase 2: Src wide local scan
            local_scan = emit_variable_scan(
                binary_fn, local_scan, custom.graph, local_scan_sizes[0]
            )

            global_scan = emit_global_scan(
                binary_fn,
                scan_src[0],
                local_scan,
                custom.graph,
                subgroup_size,
                hardware_constraint,
                local_scan_sizes[0],
                scan_dim,
            )

            # Update the users based on the global scan `reshape` results.
            for user in custom.users:
                user.update_arg(
                    custom.fx_node, global_scan[user.expanded_dims[scan_dim]]
                )

    DCE(trace)
