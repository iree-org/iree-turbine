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
from .utils.general_utils import get_fastest_index, is_valid_global_read
from .utils.graph_utils import DCE
from .utils.mapping_utils import transform_index_on_mapping
from .utils.symbol_utils import (
    subs_idxc,
)


gather_to_shared_supported_arch = ["gfx950"]


def get_write_node_consumers(read_custom):
    write_node = []

    for user in read_custom.users:
        if (
            isinstance(user, Write)
            and subs_idxc(user.memory_type.address_space) == SHARED_ADDRESS_SPACE
        ):
            write_node.append(user)

    return write_node


def gather_to_shared(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This pass enables direct memory load from global to lds without passing
    through register reducing the data movement. This instruction is supported
    only on specific architectures (gfx950).
    """

    # if get_default_arch() not in gather_to_shared_supported_arch:
    #     return

    global_read_nodes = trace.walk(is_valid_global_read)
    for read_node in global_read_nodes:
        read_custom = get_custom(read_node)
        write_consumers = get_write_node_consumers(read_custom)
        if not write_consumers:
            continue
        read_memory, read_mapping, read_type = (
            read_custom.memory,
            read_custom.mapping,
            read_custom.type,
        )

        elements_per_thread = read_custom.elements_per_thread

        for write_custom in write_consumers:
            write_memory, write_mapping, write_type = (
                write_custom.memory,
                write_custom.mapping,
                write_custom.type,
            )
            with write_custom.graph.inserting_before(write_custom.fx_node):
                write_custom.replace_all_uses_with(
                    GatherToLDS(
                        read_memory,
                        read_custom.index,
                        read_type,
                        write_memory,
                        write_memory.index,
                        write_type,
                        read_mapping,
                        write_mapping,
                        elements_per_thread,
                    ).add_to_graph(write_custom.graph)
                )
                write_custom.erase()
        read_custom.erase()
