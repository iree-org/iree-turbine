# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from ...support.logging import get_logger
from .._support.indexing import IndexingContext
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .constraints import Constraint, get_constrained_shape
from .utils import subs_idxc

logger = get_logger("turbine.wave.promotion")


def apply_promotion_pattern(custom_node: Read | Write, allocate_node: Allocate):
    match custom_node:
        case Read(memory, elements_per_thread) if get_custom(
            memory
        ).type.address_space != allocate_node.address_space:
            promoted_read = Read(
                allocate_node.fx_node, elements_per_thread
            ).add_to_graph(custom_node.graph)
            custom_node.replace_all_uses_with(promoted_read)
            with custom_node.graph.inserting_before(promoted_read):
                promoted_write = Write(
                    custom_node.fx_node, allocate_node.fx_node, elements_per_thread
                ).add_to_graph(custom_node.graph)
                custom_read = get_custom(promoted_read)
                custom_read.write_dependency = [promoted_write]
            custom_node.memory_type.address_space = GLOBAL_ADDRESS_SPACE


def promote_node(
    node: Read | Write, address_space: IndexSymbol, constraints: list[Constraint]
):
    """Promotes the given operand in the provided graph
    to the specified address space.

    The process of promotion involves allocating memory
    in the new address space and writing to the new
    memory location and subsequent uses reading from there.
    """

    assert isinstance(node, Read) or isinstance(node, Write)
    with node.graph.inserting_before(node.fx_node.next):
        constrained_shape = get_constrained_shape(node.type.symbolic_shape, constraints)
        allocate_node = Allocate(
            node.type.symbolic_shape,
            constrained_shape,
            node.type.dtype,
            address_space,
        )
        allocate_node.add_to_graph(node.graph)
        apply_promotion_pattern(node, allocate_node)


def promote_placeholders(graph: CapturedTrace, constraints: list[Constraint]):
    read_or_write_nodes = graph.walk(
        lambda node: isinstance(get_custom(node), Read)
        or isinstance(get_custom(node), Write)
    )
    for node in read_or_write_nodes:
        custom = get_custom(node)
        if not custom.memory_type:
            continue
        address_space = subs_idxc(custom.memory_type.address_space)
        if address_space == SHARED_ADDRESS_SPACE:
            promote_node(custom, address_space, constraints)
