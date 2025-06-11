# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ..lang.global_symbols import *
from ..ops.wave_ops import GatherToLDS, Write, get_custom
from ..wave.constraints import (
    Constraint,
)
from ..wave.utils.run_utils import get_default_arch
from .utils.graph_utils import DCE, is_valid_global_read
from .utils.symbol_utils import (
    subs_idxc,
)


gather_to_shared_supported_arch = ["gfx950"]


def get_write_node_info(read_node):
    read_custom = get_custom(read_node)
    write_node = None
    for user in read_custom.users:
        if (
            isinstance(user, Write)
            and subs_idxc(user.memory_type.address_space) == SHARED_ADDRESS_SPACE
        ):
            # get memory location and idx
            dst = user.memory
            dst_idx = user.get_derived_indices
            write_node = user

    return write_node, dst, dst_idx


def gather_to_shared(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function enables direct memory load from global to lds without
    passing through register reducing the data movement. This instruction
    is supported only on specific architectures (gfx950).
    """

    if get_default_arch() not in gather_to_shared_supported_arch:
        return

    global_read_nodes = trace.walk(is_valid_global_read)
    for read_node in global_read_nodes:
        custom = get_custom(read_node)
        src = custom.memory
        src_idx = custom.get_derived_indices
        element_type = custom.type.dtype
        write_node, dst, dst_idx = get_write_node_info(read_node)
        write_node.replace_all_uses_with(
            GatherToLDS(src, src_idx, dst, dst_idx, element_type)
        )

    DCE(trace)
