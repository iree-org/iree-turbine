# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
from .utils import remove_global_indexing
from .constraints import Constraint, TilingConstraint
import torch.fx as fx


def apply_shared_memory_indexing_corrections(
    trace: CapturedTrace, constraints: list[Constraint]
):
    """
    This function removes global indexing from shared memory reads and writes.
    Global indexing is an indexing that arises from Workgroup constraints
    and Tiling constraints.
    """
    tiling_constraints = [c for c in constraints if isinstance(c, TilingConstraint)]

    def is_shared_memory_read_or_write(node: fx.Node):
        custom = get_custom(node)
        if isinstance(custom, (Read, Write)):
            if custom.memory_type.address_space == SHARED_ADDRESS_SPACE:
                custom.index = remove_global_indexing(custom.index, tiling_constraints)
        return False

    trace.walk(is_shared_memory_read_or_write)
