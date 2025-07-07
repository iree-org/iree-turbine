# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence
from ..wave.constraints import (
    Constraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..lang.wave_types import IndexMapping
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
import torch.fx as fx
from collections import defaultdict
from .utils.symbol_utils import safe_subs
from .utils.general_utils import (
    get_hardware_constraint,
)
from .minimize_global_loads import (
    is_transposed_read,
    materialize_shape,
    update_write_dependencies,
)
from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..ops.wave_ops import Read, Write
from ..wave.utils.classes import LDSTransposeRead
from ..wave.utils.run_utils import get_default_arch
import logging

logger = logging.getLogger(__name__)

"""
Optimize shared -> reg for transpose using lds.tr{n} intrinsics
TODO: extend support for more variants
"""

def is_transpose_read(node: fx.Node) -> bool:
    read = get_custom(node)
    if not isinstance(read, Read):
        return False

    src_shape = read.type.symbolic_shape
    if len(src_shape) <= 1:
        return False

    return is_transposed_read(read)


def feeds_mma_instruction(write: Write) -> bool:
    write_memory = write.memory

    for user_node in write_memory.users:
        custom_user = get_custom(user_node)
        if isinstance(custom_user, Read):
            for mma_user_node in user_node.users:
                mma_custom = get_custom(mma_user_node)

                if (
                    hasattr(mma_custom, "tkw_op_name")
                    and mma_custom.tkw_op_name == "mma"
                ):
                    return True

    return False


def meets_hw_transpose_requirements(
    read: Read, write: Write, constraints: list[Constraint]
):
    if not get_default_arch() == "gfx950":
        return False

    write_memory = get_custom(write.memory)
    if write_memory.type.address_space != SHARED_ADDRESS_SPACE:
        return False

    if read.mapping_dynamic_vals:
        return False

    if read.type.dtype.bitwidth() != 8:
        return False

    if not feeds_mma_instruction(write):
        return False

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }
    materialized_shape = materialize_shape(
        constraint_tile_size, read.type.symbolic_shape
    )
    if any(s > 1 for s in materialized_shape[:-2]) or any(s <= 1 for s in materialized_shape[-2:]
    ):
        logger.info(
            f"only last 2 dims transpose is supported, got {materialized_shape}"
        )
        return False

    if materialized_shape[-2] % 16 != 0 or materialized_shape[-1] % 8 != 0:
        return False

    hardware_constraint = get_hardware_constraint(constraints)
    if hardware_constraint.threads_per_wave < 16:
        return False

    return True

def transpose_last2(shape: Sequence[IndexSymbol]) -> list[IndexSymbol]:
    return list(shape[:-2]) + [shape[-1], shape[-2]]

def mark_hardware_transpose_candidates(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    Mark shared memory allocations that can use hardware transpose.
    This is separate from in_thread_transpose optimization.
    """
    logger.info("mark_hardware_transpose_candidates")
    candidates = trace.walk(is_transpose_read)

    rw_mem_seen = set()
    new_writes = defaultdict(list)
    new_reads = defaultdict(list)

    for read in candidates:
        read = get_custom(read)
        for write in read.users:
            if not isinstance(write, Write):
                continue

            if meets_hw_transpose_requirements(read, write, constraints):
                rw_mem = (read.memory, write.memory)
                if rw_mem not in rw_mem_seen:
                    rw_mem_seen.add(rw_mem)
                    mark_hw_transpose(write, new_writes, read, new_reads, constraints)

    for old_read, new_read in new_reads.items():
        new_read_fx_node = new_read[0] 

        for user in list(old_read.users.keys()):
              custom_user = get_custom(user)
              if isinstance(custom_user, Write):
                  # Find which argument index the old read is at
                  for i, arg in enumerate(user.args):
                      if arg == old_read:
                        custom_user.update_arg(i, new_read_fx_node)
    if new_writes:
        update_write_dependencies(new_writes, trace)

def modify_index_for_full_coverage(original_index: dict) -> dict:
    modified_index = {}
    for dim, index_seq in original_index.items():
        if dim.name == 'N': 
            start_expr = index_seq.start

            modified_expr = start_expr.subs(
                sympy.Mod(THREAD_0, 16),
                sympy.Mod(THREAD_0, 32)
            )

            modified_index[dim] = IndexSequence(
                modified_expr,
                index_seq.size,
                index_seq.stride
            )
        elif dim.name == 'K': #fix out of bounds
              start_expr = index_seq.start

              old_pattern = 8 * sympy.floor(sympy.Mod(THREAD_0, 64) / 16)
              new_pattern = 8 * sympy.floor(THREAD_0 / 32) 

              modified_expr = start_expr.subs(old_pattern, new_pattern)
              modified_index[dim] = IndexSequence(modified_expr, index_seq.size, index_seq.stride)
        else:
            modified_index[dim] = index_seq

    return modified_index

def mark_hw_transpose(write: Write, new_writes: dict, read: Read, new_reads, constraints):
    with write.graph.inserting_before(write.fx_node):

        dest = get_custom(write.memory)
        dest.update_arg("hardware_transpose", LDSTransposeRead.tr8_b64)

        transpose_shape = transpose_last2(dest.distributed_shape)
        dest.update_arg("distributed_shape", transpose_shape)

        hw_write = Write(
            write.register_,
            write.memory,
            write.elements_per_thread,
            mapping=write.mapping,
            mapping_dynamic_vals=write.mapping_dynamic_vals,
        ).add_to_graph(write.graph)

        modified_index = modify_index_for_full_coverage(write.index)
        hw_write.index = modified_index
        new_writes[write.memory].append(hw_write)

        logger.info(f"Marked hardware transpose write: {hw_write}")

    with read.graph.inserting_before(read.fx_node):
                new_read = Read(
                    read.memory,
                    read.elements_per_thread,
                    mapping=read.mapping,
                    mapping_dynamic_vals=read.mapping_dynamic_vals,
                ).add_to_graph(read.graph)
                modified_index = modify_index_for_full_coverage(read.index)
                new_read.index = modified_index
                new_read.transpose = True
                new_read_custom = get_custom(new_read)
                new_read_custom.infer_type()
                if read.mapping_dynamic_vals:
                    update_read_mapping_dynamic_values(new_read_custom)
                new_reads[read.fx_node].append(new_read)