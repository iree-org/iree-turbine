# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WaveConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import (
    get_custom,
    Add,
    Allocate,
    CustomOp,
    Extract,
    Iterate,
    ExtractElement,
    Lt,
    Maximum,
    Minimum,
    Read,
    ReduceOp,
    SelectOp,
    ShuffleOp,
    Write,
    NewScalar,
)
from ..lang.global_symbols import *

from .utils.symbol_utils import subs_idxc
from .utils.graph_utils import DCE
from .utils.general_utils import all_equal
import torch.fx as fx
import math
from typing import Callable
import iree.turbine.kernel.lang as tkl

TKW_COMBINER = {"sum": Add, "max": Maximum, "min": Minimum}
IDENTITY = {"sum": 0.0, "max": -1e6, "min": 1e6}

_MAX_READ_BITWIDTH = 128


def determine_shuffle_config(
    index: dict[IndexSymbol, IndexSequence],
    reduction_dim: IndexSymbol,
    vector_shapes: dict[IndexSymbol, int],
    subgroup_size: int,
    induction_vars: list[IndexSymbol],
):
    """
    This function determines the cluster size and stride for a given index.
    The cluster size specifies the number of threads that participate in a shuffle.
    The cluster stride specifies the stride between the threads. In order to
    determine the cluster stride, we do a binary search on the start value of the
    index sequence.

    """
    access_pattern = index[reduction_dim]

    # Since we are only concerned with what happens within a subgroup,
    # we can ignore the TID_1 and TID_2 components of the index. We can
    # also ignore the GPR_NUM since we can assume we are only dealing with the
    # same GPR_NUM. We ignore the workgroup indices and induction variables as well.
    # Finally, we substitute in all variables that are known in the indexing context.
    ignore = [
        THREAD_1,
        THREAD_2,
        GPR_NUM,
        WORKGROUP_0,
        WORKGROUP_1,
        WORKGROUP_2,
    ] + induction_vars
    offset = access_pattern.start.subs({k: 0 for k in ignore})
    offset = subs_idxc(offset)
    offset_table = [offset.subs({THREAD_0: i}) for i in range(subgroup_size)]
    unique_offsets = list(dict.fromkeys(offset_table))
    # The cluster size represents the number of unique threads that are participating in a shuffle.
    # We can obtain this information by just computing the number of unique entries in the offset table.
    cluster_size = len(unique_offsets)
    thread_ids = []
    for thread_offset in unique_offsets:
        thread_ids.append(offset_table.index(thread_offset))
    cluster_stride = [x - y for x, y in zip(thread_ids[1:], thread_ids[:-1])]
    assert all_equal(cluster_stride), f"Cluster stride must be equal across threads."
    return cluster_size, cluster_stride[0]


def get_graph_node(custom: CustomOp, graph: fx.Graph) -> fx.Node:
    custom.add_to_graph(graph)
    custom = custom.fx_node
    return custom


def emit_sources_reduction(
    binary_fn: Callable, src: list[fx.Node], graph: fx.Graph
) -> fx.Node:
    """
    Does reduction over a list of fx.Node variables by applying binary_fn on them.
    """
    init = src[0]
    for i in range(1, len(src)):
        init = get_graph_node(binary_fn(init, src[i]), graph)
    init.index = src[0].index
    return init


def emit_variable_reduction(
    binary_fn: Callable, src: fx.Node, graph: fx.Graph, local_reduction_size: int
) -> fx.Node:
    """
    Does reduction over a singular fx.Node variable.
    """
    init = get_graph_node(Extract(src, [0]), graph)
    for i in range(1, local_reduction_size):
        cur_slice = get_graph_node(Extract(src, [i]), graph)
        init = get_graph_node(binary_fn(init, cur_slice), graph)
    return init


def emit_local_reduction(
    binary_fn: Callable,
    reduction_src: list[fx.Node],
    graph: fx.Graph,
    local_reduction_size,
) -> fx.Node:
    """
    Does reduction over all the element carried along by ReductionOp at local
    thread/SIMT level. This is done by reducing expanded sources combining them
    into single variable, and then reducing that variable into a scalar.
    """
    src_reduction = emit_sources_reduction(binary_fn, reduction_src, graph)
    local_reduction = emit_variable_reduction(
        binary_fn, src_reduction, graph, local_reduction_size
    )
    return local_reduction


def emit_scalarized_local_reduction(
    binary_fn: Callable,
    reduction_src: list[fx.Node],
    graph: fx.Graph,
    local_reduction_size,
) -> fx.Node:
    """
    Special case of local reduction wher we try to scalarize/get rid of most vector ops.
    this is useful for maximum, to expose more opportunities for v_max3_f32,
    We do this by first reducing the sources(scalar/iterative manner), and then
    reducing all the "reduced" args/source.
    e.g we transform from:

    %source_reduce = arith.maximumf %lhs, %rhs : vector<16xf32>
    %local_reduce = vector.reduction<maximumf>, %src_reduce : f32 from vector<16xf32>

    into:

    %local_lhs_reduce = vector.reduction<maximumf>, %lhs : f32 from vector<16xf32>
    %local_rhs_reduce = vector.reduction<maximumf>, %rhs : f32 from vector<16xf32>
    %local_src_reduce = arith.maximumf %local_lhs_reduce, %local_rhs_reduce : f32
    """
    locally_reduced_sources = [
        emit_variable_reduction(binary_fn, arg, graph, local_reduction_size)
        for arg in reduction_src
    ]
    local_reduction = emit_sources_reduction(binary_fn, locally_reduced_sources, graph)
    return local_reduction


def emit_global_reduction(
    binary_fn: Callable,
    src: fx.Node,
    graph: fx.Graph,
    subgroup_size: int,
    cluster_size: int,
    cluster_stride: int,
) -> fx.Node:
    """
    Reduce data across threads in a warp by doing butterfly shuffle.
    """
    init = src
    num_steps = int(math.log2(float(cluster_size)))
    for _ in range(num_steps):
        shuffle_val = ShuffleOp(init, cluster_stride, subgroup_size)
        shuffle_node = get_graph_node(shuffle_val, graph)
        init = get_graph_node(binary_fn(init, shuffle_node), graph)
        cluster_stride <<= 1
    return init


def decompose_reduce_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    The lowering for multi_reduction is done in two steps:
      1. Source Reduce: Each thread reduce locally all it's sources.
      2. Local Reduce: Each thread reduces all elements carried by it along
         the reduction dimensions.
      3. Thread Reduce: Each thread reduces result of step 2 across threads
         by doing a butterfly shuffle.
      4. Accumulator Reduce: Each thread reduces it's intermediate reduced
         results with the accumulator it holds.
    """
    # Get reducte nodes.
    reduce_nodes = trace.walk(lambda node: isinstance(get_custom(node), ReduceOp))
    if not reduce_nodes:
        return

    # Setup constraints
    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    induction_vars = [
        c.induction_var for c in constraints if isinstance(c, TilingConstraint)
    ]

    wave_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WaveConstraint)
    }
    workgroup_constraint_map = {
        c.dim: c for c in constraints if isinstance(c, WorkgroupConstraint)
    }
    subgroup_size = hardware_constraint.threads_per_wave
    for node in reduce_nodes:
        custom = get_custom(node)
        with custom.graph.inserting_before(custom.fx_node):
            reduction_src, reduction_acc, reduction_dim, reduce_block = node.args
            binary_fn = TKW_COMBINER[custom.tkw_op_name]
            if reduction_dim is None:
                raise ValueError(
                    "No reduction dim specified, please specify a reduction dim."
                )
            if not isinstance(reduction_src, (list, tuple)):
                reduction_src = [reduction_src]

            # Local Reduce
            src_fastest_dims = [
                get_custom(arg).type.symbolic_shape[-1] for arg in reduction_src
            ]
            if not all_equal(src_fastest_dims):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same fastest dim."
                )
            if reduction_dim is not src_fastest_dims[0]:
                raise NotImplementedError(
                    "Only implemented reduction on fastest dimension."
                )

            get_thread_shape = lambda index: max(
                subs_idxc(x.size) for x in index.values()
            )
            local_reduce_sizes = []
            for arg in reduction_src:
                try:
                    op = get_custom(arg)

                    thread_shape = get_thread_shape(op.index)
                    local_reduce_sizes.append(thread_shape)
                except Exception as e:
                    index_str = "\n".join(f"{k}: {v}" for k, v in op.index.items())
                    raise RuntimeError(
                        f"Error in decompose_reduce_ops: {arg} with index\n"
                        f"{index_str}\n{reduction_src=}\n{reduction_acc=}\n{reduction_dim=}"
                    ) from e

            if not all_equal(local_reduce_sizes):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same local reduce size."
                )
            if binary_fn == Maximum:
                local_reduction = emit_scalarized_local_reduction(
                    binary_fn, reduction_src, custom.graph, local_reduce_sizes[0]
                )
            else:
                local_reduction = emit_local_reduction(
                    binary_fn, reduction_src, custom.graph, local_reduce_sizes[0]
                )

            if (
                reduction_acc is not None
                and get_custom(local_reduction).type.symbolic_shape
                != get_custom(reduction_acc).type.symbolic_shape
            ):
                raise RuntimeError(
                    "Local reduction and accumulator reduction must have same shape."
                    f"\nlocal_reduction: {get_custom(local_reduction).type.symbolic_shape}"
                    f"\nreduction_acc: {get_custom(reduction_acc).type.symbolic_shape}"
                    f"\nlocal_reduction: {get_custom(local_reduction)}"
                    f"\nreduction_acc: {get_custom(reduction_acc)}"
                    f"\n{custom}"
                )
            # Global Reduce
            cluster_size, cluster_stride = determine_shuffle_config(
                reduction_src[0].index,
                reduction_dim,
                node.vector_shapes,
                subgroup_size,
                induction_vars,
            )
            global_reduction = emit_global_reduction(
                binary_fn,
                local_reduction,
                custom.graph,
                subgroup_size,
                cluster_size,
                cluster_stride,
            )

            # Local Accumulator Reduce
            final_reduction = global_reduction
            if reduction_acc is not None:
                final_reduction = get_graph_node(
                    binary_fn(reduction_acc, global_reduction), custom.graph
                )

            if reduce_block:
                # compute num_warps to reduce across
                num_reduction_waves = int(
                    workgroup_constraint_map[reduction_dim].tile_size
                    // wave_constraint_map[reduction_dim].tile_size
                )
                if num_reduction_waves > subgroup_size:
                    raise NotImplementedError(
                        "The 2nd stage butterfly shuffle reduces the"
                        "the reduction outputs from all the wave. Hence, can only handle at most "
                        "threads_per_wave number of warps."
                    )
                # Reduction inter-warp by writing then reading to/from shared memory.
                allocate_node = Allocate(
                    (reduction_dim,),
                    (num_reduction_waves,),
                    node.type.dtype,
                    SHARED_ADDRESS_SPACE,
                ).add_to_graph(custom.graph)
                write = Write(final_reduction, allocate_node, 1).add_to_graph(
                    custom.graph
                )

                # Getting wave_id of the wave-dim that is being used for reduction.
                # This is useful because depending on wg_dim we choose to reduce on
                # the wave_id and size of partial output will be different.
                # e.g wave_per_block = (4, 2, 1)
                # if reduc_dim on wg_dim=0, then we will only have 4 partial output.
                # if reduc_dim on wg_dim=1, then we will only have 2 partial output.
                reduction_wg_dim = workgroup_constraint_map[reduction_dim].workgroup_dim
                reduction_wave_id = hardware_constraint.wave_id[reduction_wg_dim]
                write.index = {reduction_dim: IndexSequence(reduction_wave_id, 1, 1)}

                # Do a 2nd stage warp reduction on the results from different warp.
                read = Read(
                    allocate_node,
                    elements_per_thread=1,
                    _write_dependency=[write],
                ).add_to_graph(custom.graph)
                read.index = {
                    reduction_dim: IndexSequence(hardware_constraint.lane_id, 1, 1)
                }
                warp_partial_res = get_graph_node(
                    ExtractElement(read, [0]), custom.graph
                )
                warp_partial_res.type = node.type.dtype

                # TODO: Add handler for newScalar to handle IndexExpr and constant.
                lane_id_reg = get_graph_node(
                    NewScalar(hardware_constraint.lane_id, tkl.i32), custom.graph
                )
                num_reduction_waves_regs = get_graph_node(
                    NewScalar(num_reduction_waves, tkl.i32), custom.graph
                )
                id_reg = get_graph_node(
                    NewScalar(IDENTITY[custom.tkw_op_name], node.type.dtype),
                    custom.graph,
                )
                cond = get_graph_node(
                    Lt(lane_id_reg, num_reduction_waves_regs), custom.graph
                )
                masked_value = get_graph_node(
                    SelectOp(cond, warp_partial_res, id_reg), custom.graph
                )

                final_reduction = emit_global_reduction(
                    binary_fn,
                    masked_value,
                    custom.graph,
                    subgroup_size,
                    cluster_size=64,
                    cluster_stride=1,
                )

            # Replace all uses with global reduction
            custom.replace_all_uses_with(final_reduction)

    DCE(trace)
