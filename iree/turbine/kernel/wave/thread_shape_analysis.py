# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...support.logging import get_logger
from iree.turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .utils import capture_forward_slice, capture_backward_slice, subs_idxc

logger = get_logger("turbine.wave.thread_shape_analysis")

################################################################
# Index/Symbol and Thread size helper fn and data structure
#################################################################


@dataclass(order=True)
class DimSize:
    dim: IndexSymbol
    size: int

    def __hash__(self):
        return hash((self.dim, self.size))


def get_dim_sizes(indices: list[IndexSequence]):
    dims = frozenset(
        [DimSize(dim, subs_idxc(seq.size)) for dim, seq in indices.items()]
    )
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


#################################################################
# Anchor Indicies and Conflict resolution helpers
#################################################################

anchorOpTypes = (Read, Write, MMA, ReduceOp, Reshape)
noHandleTypes = (Placeholder, Output, ExtractSlice, Allocate)
legalSubtypes = (IterArg,)
nonPropagatableTypes = anchorOpTypes + noHandleTypes


def is_anchor_op(node: fx.Node):
    return isinstance(get_custom(node), anchorOpTypes)


def propagatable_op(node: fx.Node):
    custom_node = get_custom(node)
    return not isinstance(custom_node, nonPropagatableTypes) or isinstance(
        custom_node, legalSubtypes
    )


def handle_binaryop_conflict(custom_node: CustomOp) -> list[fx.Node]:
    """
    This function will attempt to resolve binaryOp conflicts
    by inserting broadcastOp. It will then propagate the resolutions,
    and return the list of fx.Nodes that we have resolved.
    """

    # Analyze if we can resolve conflict with broadcast.
    lhs = get_custom(custom_node.lhs)
    rhs = get_custom(custom_node.rhs)
    lhs_dim_set = set(lhs.type.symbolic_shape)
    rhs_dim_set = set(rhs.type.symbolic_shape)
    if lhs_dim_set == rhs_dim_set:
        # Could be caused by consumers(likely also binaryOp) of this node.
        return []
    if lhs_dim_set.isdisjoint(rhs_dim_set):
        raise ValueError("Cannot broadcast if lhs and rhs has disjointed shapes.")
    # Determine the correct indexSize for binaryOp and insert broadcasting.
    dst_op = lhs if lhs_dim_set > rhs_dim_set else rhs
    broadcast_idx, broadcast_src = (1, rhs) if lhs_dim_set > rhs_dim_set else (0, lhs)
    with custom_node.graph.inserting_before(custom_node.fx_node):
        broadcast = Broadcast(broadcast_src.fx_node, dst_op.type).add_to_graph(
            custom_node.graph
        )
        custom_broadcast = get_custom(broadcast)
        custom_broadcast.vector_shapes = broadcast_src.vector_shapes
        custom_broadcast.anchor = broadcast_src.anchor
        custom_node.update_arg(broadcast_idx, custom_broadcast.fx_node)
    propagated_resolutions = capture_forward_slice(
        custom_broadcast.fx_node, propagatable_op
    )
    for node in propagated_resolutions:
        get_custom(node).index = dst_op.index
    resolved_resolutions = capture_backward_slice(
        custom_broadcast.fx_node, propagatable_op
    )
    return propagated_resolutions.union(resolved_resolutions)


# Returns True iff all conflicts are handled succesfully.
def handle_conflicts(conflicted_ops: set[CustomOp]):
    cummulative_resolved = set()
    for conflict in conflicted_ops:
        custom = get_custom(conflict)
        if isinstance(custom, BinaryPyOp):
            resolved_ops = handle_binaryop_conflict(custom)
            cummulative_resolved = cummulative_resolved.union(resolved_ops)
        else:
            continue
    # Superset because path/cumulative resolved includes resolution helper ops
    # such as broadcast.
    all_conflicts_resolved = cummulative_resolved.issuperset(conflicted_ops)
    return all_conflicts_resolved


###############################################################################
# Main pass
#####################################################################


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
    anchor_ops = trace.walk(is_anchor_op)
    thread_size_to_ops: dict[frozenset[DimSize], set[CustomOp]] = {}
    for anchor_op in anchor_ops:
        custom = get_custom(anchor_op)
        index_sizes = get_custom_dim_sizes(custom)
        if isinstance(custom, Read):
            fwd_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(fwd_slice)
        elif isinstance(custom, ReduceOp):
            fwd_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            bwd_slice = set()
            if custom.init != None and not isinstance(
                get_custom(custom.init), ReduceOp
            ):
                bwd_slice = capture_backward_slice(custom.init, propagatable_op)
            reduce_dims = frozenset(
                [DimSize(dim, 1) for dim in custom.index.keys() if dim != custom.dim]
            )
            thread_size_to_ops[reduce_dims] = (
                thread_size_to_ops.get(reduce_dims, set([]))
                .union(fwd_slice)
                .union(bwd_slice)
            )
        elif isinstance(custom, Write):
            bwd_slice = capture_backward_slice(custom.fx_node, propagatable_op)
            thread_size_to_ops[index_sizes] = thread_size_to_ops.get(
                index_sizes, set([])
            ).union(bwd_slice)
        elif isinstance(custom, MMA):
            lhs_bwd_slice = set([custom.lhs])
            if propagatable_op(custom.lhs):
                lhs_bwd_slice = capture_backward_slice(custom.lhs, propagatable_op)
            rhs_bwd_slice = set([custom.rhs])
            if propagatable_op(custom.rhs):
                rhs_bwd_slice = capture_backward_slice(custom.rhs, propagatable_op)
            acc_slice = capture_forward_slice(custom.fx_node, propagatable_op)
            if not isinstance(get_custom(custom.acc), MMA):
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
        elif isinstance(custom, Reshape):
            # The reshape op acts like a barrier for the MMA preventing
            # the mma from propagating the thread shapes of its reshaped
            # operands backwards.
            bwd_size = get_dim_sizes(custom.args.index)
            bwd_slice = capture_backward_slice(custom.args, propagatable_op)
            thread_size_to_ops[bwd_size] = thread_size_to_ops.get(
                bwd_size, set([])
            ).union(bwd_slice)

    # Go through each index-size buckets, and apply the index-size to ops in the bucket.
    cummulative_set = set()
    for target_index_size, target_ops in thread_size_to_ops.items():
        # Try to handle conflicts and remove from target set if successfully handled.
        if not cummulative_set.isdisjoint(target_ops):
            conflicted_ops = cummulative_set.intersection(target_ops)
            if handle_conflicts(conflicted_ops) == False:
                raise NotImplementedError("Failed to handle conflicting thread shape.")
            target_ops = target_ops.difference(conflicted_ops)
        cummulative_set = cummulative_set.union(target_ops)
        # Set target ops's indexSize to be the determined from analysis.
        for user in target_ops:
            custom_user = get_custom(user)
            set_index_size(custom_user, target_index_size)
