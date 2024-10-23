# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexingContext, IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import (
    get_custom,
    Add,
    Maximum,
    ReduceOp,
    ShuffleOp,
    CustomOp,
    Extract,
    Reduction,
)

from .utils import DCE, subs_idxc, all_equal
import torch.fx as fx
import math
from typing import Callable

TKW_COMBINER = {"sum": Add, "max": Maximum}


def get_graph_node(custom: CustomOp, graph: fx.Graph):
    custom.add_to_graph(graph)
    custom = custom.fx_node
    return custom


def emit_sources_reduction(
    binary_fn: Callable, src: list[fx.Node], graph: fx.Graph
) -> fx.Node:
    init = src[0]
    for i in range(1, len(src)):
        init = get_graph_node(binary_fn(init, src[i]), graph)
    init.index = src[0].index
    return init


def emit_local_reduction(
    binary_fn: Callable, src: fx.Node, graph: fx.Graph, local_reduction_size: int
) -> fx.Node:
    init = get_graph_node(Extract(src, [0]), graph)
    for i in range(1, local_reduction_size):
        cur_slice = get_graph_node(Extract(src, [i]), graph)
        init = get_graph_node(binary_fn(init, cur_slice), graph)
    return init


def emit_global_reduction(
    binary_fn: Callable, src: fx.Node, graph: fx.Graph, subgroup_size: int
) -> fx.Node:
    init = src
    num_steps = int(math.log2(float(subgroup_size)))
    for i in range(num_steps):
        shuffle_offset = 2**i
        shuffle_val = ShuffleOp(init, shuffle_offset, subgroup_size)
        shuffle_node = get_graph_node(shuffle_val, graph)
        init = get_graph_node(binary_fn(init, shuffle_node), graph)
    return init


def decompose_reduce_ops(
    trace: CapturedTrace,
    constraints: list[Constraint],
    index_map: dict[IndexSymbol, int],
):
    """
    The lowering for multi_reduction is done in two steps:
      1. Source Reduce: Each thread reduce locally all it's sources.
      2. Local Reduce: Each thread reduces all elements carried by it along
         the reduction dimensions.
      3. Thread Reduce: Each thread reduces result of step 1 across threads
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
    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }
    subgroup_size = hardware_constraint.threads_per_wave
    for node in reduce_nodes:
        custom = get_custom(node)
        with custom.graph.inserting_before(custom.fx_node):
            reduction_src, reduction_acc, reduction_dim = node.args
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
            local_reduce_sizes = [
                get_thread_shape(get_custom(arg).index) for arg in reduction_src
            ]
            if not all_equal(local_reduce_sizes):
                raise NotImplementedError(
                    "NYI: Expect all reduce_src to have same local reduce size."
                )
            src_reduction = emit_sources_reduction(
                binary_fn, reduction_src, custom.graph
            )
            local_reduction = emit_local_reduction(
                binary_fn, src_reduction, custom.graph, local_reduce_sizes[0]
            )

            # Global Reduce
            global_reduction = emit_global_reduction(
                binary_fn, local_reduction, custom.graph, subgroup_size
            )

            # Local Accumulator Reduce
            final_reduction = global_reduction
            if reduction_acc is not None:
                final_reduction = get_graph_node(
                    binary_fn(reduction_acc, global_reduction), custom.graph
                )

            # Replace all uses with global reduction
            custom.replace_all_uses_with(final_reduction)

    DCE(trace)
