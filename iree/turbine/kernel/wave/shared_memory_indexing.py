# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, Write, MMA, get_custom
from ..lang.global_symbols import *
from .utils import remove_global_indexing, align_index_vars
from .constraints import Constraint, TilingConstraint
import torch.fx as fx


def is_shared_mem_access(custom: "CustomOp") -> bool:
    return custom.memory_type.address_space == SHARED_ADDRESS_SPACE


def apply_shared_memory_indexing_corrections(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    This function removes global indexing from shared memory reads and writes.
    Global indexing is an indexing that arises from Workgroup constraints
    and Tiling constraints.
    """

    def is_shared_memory_read_or_write(node: fx.Node):
        custom = get_custom(node)
        if isinstance(custom, (Read, Write)) and is_shared_mem_access(custom):
            custom.index = remove_global_indexing(custom.index, constraints)
        return False

    trace.walk(is_shared_memory_read_or_write)


def align_index_sizes(trace: CapturedTrace, constraints: list[Constraint]):
    """
    Adjust ops index sizes to WG/Tile size, so shared mem ops never need to
    do partial read/writes.
    """

    def need_align(node: fx.Node):
        custom = get_custom(node)
        if isinstance(custom, (Read, Write)) and is_shared_mem_access(custom):
            custom.index = align_index_vars(custom.index, constraints)
        elif isinstance(custom, MMA):
            custom.index = align_index_vars(custom.index, constraints)
        return False

    trace.walk(need_align)
