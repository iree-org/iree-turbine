# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from .._support.dtype import DataType
from .._support.tracing import CapturedTrace
from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .utils import print_graph, get_users, subs_idxc, is_shared_read, is_shared_write
from ..ops.wave_ops import get_custom, CustomOp, Read, Write, wave_xor
from dataclasses import dataclass
from .._support.indexing import IndexExpr, IndexSequence, index_symbol
from ..lang.global_symbols import ELEMENT_INDEX


@dataclass
class SwizzleConfig:
    num_banks: int = 32
    simd_lanes: int = 16
    bank_width: int = 32  # in bits


@dataclass
class DimShape:
    dim: int
    shape: IndexExpr


def swizzle_index(
    shmem_write: Write, shmem_reads: list[Read], already_swizzled: set[Read]
):
    """
    Swizzle the index along the contiguous dimension to avoid bank conflicts.
    """
    dim_shape = [
        DimShape(dim, subs_idxc(x))
        for dim, x in zip(
            shmem_write.indexing_dims, get_custom(shmem_write.memory).distributed_shape
        )
    ]
    inner_dim = dim_shape[-1]
    outer_dim = [x for x in dim_shape if x != inner_dim and x.shape > 1]
    assert len(outer_dim) == 1, f"Expected 1 outer dimension, got {len(outer_dim)}"
    outer_dim = outer_dim[0]

    swizzle_config = SwizzleConfig()
    bitwidth = shmem_write.type.dtype.bitwidth()
    elems_per_bank = (swizzle_config.num_banks * swizzle_config.bank_width) // bitwidth
    per_phase = max(1, elems_per_bank // inner_dim.shape)
    vec_size = subs_idxc(shmem_write.elements_per_thread)
    max_phase = max(
        min(
            swizzle_config.simd_lanes // per_phase,
            inner_dim.shape // vec_size,
        ),
        1,
    )

    swizzle_fn = lambda row, col: wave_xor(
        (col // vec_size), (row // per_phase) % max_phase
    ) * vec_size + (col % vec_size)

    def update_row_index(
        op: CustomOp, inner_dim: DimShape, row: IndexExpr
    ) -> IndexExpr:
        # If this is a gather, then we need to add a new symbol to represent
        # the fact that the column index will be different for each
        # value of the row index.
        if op.index[inner_dim.dim].size == 1:
            row += ELEMENT_INDEX
        return row

    row = shmem_write.index[outer_dim.dim].start
    col = shmem_write.index[inner_dim.dim].start
    row = update_row_index(shmem_write, inner_dim, row)
    shmem_write.index[inner_dim.dim].start = swizzle_fn(row, col)
    for read in shmem_reads:
        if read in already_swizzled:
            continue
        row = read.index[outer_dim.dim].start
        col = read.index[inner_dim.dim].start
        row = update_row_index(read, inner_dim, row)
        read.index[inner_dim.dim].start = swizzle_fn(row, col)
        already_swizzled.add(read)


def bytes_transferred(shmem_op: Read | Write):
    """
    Return the number of bytes transferred by the shared memory operation.
    """
    elems_per_thread = shmem_op.elements_per_thread
    bitwidth = shmem_op.type.dtype.bitwidth()
    return subs_idxc(elems_per_thread * bitwidth)


def swizzle_access_patterns(
    shmem_write: Write, shmem_reads: list[Read], already_swizzled: set[Read]
):
    """
    Swizzle the read/write index depending on the bytes being read/written.
    """
    write_bytes = bytes_transferred(shmem_write)
    read_bytes = [bytes_transferred(x) for x in shmem_reads]
    print(write_bytes, read_bytes)

    # if write_bytes != 128 or not all(x == write_bytes for x in read_bytes):
    #    return

    swizzle_index(shmem_write, shmem_reads, already_swizzled)


def swizzle_shared_memory(trace: CapturedTrace, constraints: list[Constraint]):
    """
    Swizzle shared memory read and write accesses to avoid bank conflicts.
    """
    shmem_writes = trace.walk(lambda x: is_shared_write(get_custom(x)))
    already_swizzled = set()
    for shmem_write in shmem_writes:
        shmem_reads, _ = get_users(shmem_write, None)
        swizzle_access_patterns(
            get_custom(shmem_write),
            [get_custom(x) for x in shmem_reads],
            already_swizzled,
        )
