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
class DimSize:
    dim: IndexSymbol
    size: int

    def __hash__(self):
        return hash((self.dim, self.size))


def get_dim_sizes(indices: list[IndexSequence]):
    dims = frozenset([DimSize(dim, seq.size) for dim, seq in indices.items()])
    return dims


def get_custom_dim_sizes(custom: CustomOp):
    return get_dim_sizes(custom.index)


def set_index_size(custom: CustomOp, target_dim_sizes: list[DimSize]):
    for target in target_dim_sizes:
        if target.dim not in custom.index:
            raise NotImplementedError(
                "NYI: Handle when source target index size is not found in target/user index."
            )
        custom.index[target.dim].size = target.size


def determine_thread_shapes(trace: CapturedTrace):
    """
    This function does analysis and propagation of thread shape. It does by such:
    1. Look for "anchor" ops who has information of it's elem_per_thread.
    2. Do a forward/backward slice on these anchor ops to get ops that
    who's shapes depends on these anchor ops.
    3. We bucket these ops to Variadic(Index->elem_per_thread) mapping.
    4. At every bucket of (index -> elem_per_thread), we apply these information
       by updating their indexSequence size.

    We stored the buckets above in a variable/dict called `thread_size_to_ops`.

    `thread_size_to_ops` is a dict that uses thread_shapes as key and for every
    key/thread_shape will map to a set of fx.nodes that needs to have that
    thread_shape in it's indexSequence.

    `thread_shapes` is used to store thread_size at every dimension that the op
    cares about. We use a frozenset[DimSize] to represent it, where  DimSize
    is essentially a pair<dimension: IndexSymbol, thread_size: int>. we are using
    frozen_set since we do not care about the order of dims for the shape/size
    propagation.

    We use sets[CustomOp] to represent the values of `thread_size_ops` S.T we can
    easily find any conflicting of index using set operations and handle/resolve it
    if required.

    For better illustration, here's an example:
    Kernel:
        imm = tkw.mul(x, y)
        lhs = tkw.neg(imm)
        a = tkw.mma(lhs, rhs, acc)
        b = tkw.exp2(a)
    Anchors:
        mma.lhs: {IndexSize(index=M, size=1), IndexSize(index=K, size=4)}
        mma.rhs: {IndexSize(index=K, size=4), IndexSize(index=N, size=1)}
        mma.acc: {IndexSize(index=M, size=4), IndexSize(index=N, size=1)}
    Bucket Entry:
        thread_sizes_to_ops[frozenset({IndexSize(index=M, size=1), IndexSize(index=K, size=4)}] = set(lhs, imm, x, y)
        thread_sizes_to_ops[frozenset({IndexSize(index=M, size=4), IndexSize(index=N, size=1)}] = set(acc, exp2_0)
        thread_sizes_to_ops[frozenset({IndexSize(index=K, size=4), IndexSize(index=N, size=1)}] = set(rhs, ...)

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
    thread_size_to_ops: dict[frozenset[DimSize], set[CustomOp]] = {}
    for anchor_op in anchor_ops:
        custom = get_custom(anchor_op)
        index_sizes = get_custom_dim_sizes(custom)
        if isinstance(custom, (Read, ReduceOp)):
            fwd_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(fwd_slice)
        elif isinstance(custom, Write):
            bwd_slice = capture_backward_slice(custom.fx_node, propagatable_op)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(bwd_slice)
        elif isinstance(custom, MMA):
            lhs_bwd_slice = capture_backward_slice(custom.lhs, propagatable_op)
            rhs_bwd_slice = capture_backward_slice(custom.rhs, propagatable_op)
            acc_slice = capture_forward_slice(custom.acc, propagatable_op)
            acc_slice = acc_slice.union(
                capture_backward_slice(custom.acc, propagatable_op)
            )
            acc_index = get_dim_sizes(custom.acc_index)
            lhs_index = get_dim_sizes(custom.lhs_index)
            rhs_index = get_dim_sizes(custom.rhs_index)
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
