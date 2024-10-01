# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...support.logging import get_logger
from shark_turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .utils import capture_forward_slice, capture_backward_slice

logger = get_logger("turbine.wave.thread_shape_analysis")


@dataclass(order=True)
class IndexSize:
    index: IndexSymbol
    size: int

    def __hash__(self):
        return hash((self.index, self.size))


def get_index_sizes(indices: list[IndexSequence]):
    dims = frozenset([IndexSize(index, seq.size) for index, seq in indices.items()])
    return dims


def get_custom_index_sizes(custom: CustomOp):
    return get_index_sizes(custom.index)


def set_index_size(custom: CustomOp, target_index_sizes: list[IndexSize]):
    for target in target_index_sizes:
        if target.index not in custom.index:
            raise NotImplementedError(
                "NYI: Handle when source target index size is not found in target/user index."
            )
        custom.index[target.index].size = target.size


# Function called on op post propagation for extra processing/handling.
def post_propagation(custom: CustomOp, target_index_sizes: list[IndexSize]):
    if isinstance(custom, IterArg):
        init_args = custom.parent_op().init_args[custom.get_iter_idx()]
        set_index_size(get_custom(init_args), target_index_sizes)


def determine_thread_shapes(trace: CapturedTrace):
    """
    This function does analysis and propagation of thread shape. It does by such:
    1. Look for "anchor" ops who has information of it's elem_per_thread.
    2. Do a forward/backward slice on these anchor ops to get ops that
    who's shapes depends on these anchor ops.
    3. We bucket these ops to Variadic(Index->elem_per_thread) mapping.
    4. At every bucket of (index -> elem_per_thread), we apply these information
       by updating their indexSequence size.
    """

    # Anchor ops are ops who's thread shape are predetermined.
    anchorOpTypes = (Read, Write, MMA, ReduceOp)
    noHandleTypes = (Placeholder, Output, ExtractSlice, Allocate)
    nonPropagatableTypes = anchorOpTypes + noHandleTypes

    def is_anchor_op(node: fx.Node):
        return isinstance(get_custom(node), anchorOpTypes)

    def propagatable_op(node: fx.Node):
        return not isinstance(get_custom(node), nonPropagatableTypes)

    anchor_ops = trace.walk(is_anchor_op)
    thread_size_to_ops: dict[IndexSymbol, set[CustomOp]] = {}
    for anchor_op in anchor_ops:
        custom = get_custom(anchor_op)
        index_sizes = get_custom_index_sizes(custom)
        if isinstance(custom, (Read, ReduceOp)):
            fwd_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            fwd_slice.remove(custom.fx_node)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(fwd_slice)
        elif isinstance(custom, Write):
            bwd_slice = capture_backward_slice(custom.fx_node, propagatable_op)
            bwd_slice.remove(custom.fx_node)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(bwd_slice)
        elif isinstance(custom, MMA):
            lhs_bwd_slice = capture_backward_slice(custom.lhs, propagatable_op)
            rhs_bwd_slice = capture_backward_slice(custom.rhs, propagatable_op)
            acc_slice = capture_forward_slice(custom.acc, propagatable_op)
            acc_slice.union(capture_backward_slice(custom.acc, propagatable_op))
            acc_index = get_index_sizes(custom.acc_index)
            lhs_index = get_index_sizes(custom.lhs_index)
            rhs_index = get_index_sizes(custom.rhs_index)
            thread_size_to_ops[acc_index] = thread_size_to_ops.get(
                acc_index, set([])
            ).union(acc_slice)
            thread_size_to_ops[lhs_index] = thread_size_to_ops.get(
                lhs_index, set([])
            ).union(lhs_bwd_slice)
            thread_size_to_ops[rhs_index] = thread_size_to_ops.get(
                rhs_index, set([])
            ).union(rhs_bwd_slice)

    # Go through each index-size buckets, and apply the index-size to ops in the bucket.
    cummulative_set = set()
    for target_index_size, target_ops in thread_size_to_ops.items():
        # Ensure that we do not have any conflicts.
        if not cummulative_set.isdisjoint(target_ops):
            raise NotImplementedError("NYI: Handling of conflicting thread shape.")
        cummulative_set = cummulative_set.union(target_ops)
        for user in target_ops:
            custom_user = get_custom(user)
            set_index_size(custom_user, target_index_size)
            post_propagation(custom_user, target_index_size)
