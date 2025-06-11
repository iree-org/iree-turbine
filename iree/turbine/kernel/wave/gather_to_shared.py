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
from .utils.general_utils import is_valid_global_read
from .utils.graph_utils import DCE
from .utils.symbol_utils import (
    subs_idxc,
)


gather_to_shared_supported_arch = ["gfx950"]


def get_write_node_info(read_custom):
    write_node, write_memory, write_idx = [], [], []

    for user in read_custom.users:
        if (
            isinstance(user, Write)
            and subs_idxc(user.memory_type.address_space) == SHARED_ADDRESS_SPACE
        ):
            write_node.append(user)
            write_memory.append(user.memory)
            write_idx.append(user.get_derived_indices[0])

    return write_node, write_memory, write_idx


def gather_to_shared(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This pass enables direct memory load from global to lds without passing
    through register reducing the data movement. This instruction is supported
    only on specific architectures (gfx950).
    """

    if get_default_arch() not in gather_to_shared_supported_arch:
        return

    global_read_nodes = trace.walk(is_valid_global_read)
    for read_node in global_read_nodes:
        read_custom = get_custom(read_node)
        src = read_custom.memory
        src_idx = read_custom.get_derived_indices[0]
        element_type = read_custom.type.dtype
        write_node, write_memory, write_idx = get_write_node_info(read_custom)
        if not write_node:
            continue
        for (dst_node, dst_memory, dst_idx) in zip(write_node, write_memory, write_idx):
            with dst_node.graph.inserting_before(dst_node.fx_node):
                dst_node.replace_all_uses_with(
                    GatherToLDS(
                        src, src_idx, dst_memory, dst_idx, element_type
                    ).add_to_graph(dst_node.graph)
                )

    DCE(trace)
