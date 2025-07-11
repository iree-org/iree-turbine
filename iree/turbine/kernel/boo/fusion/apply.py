# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Sequence

from torch import fx

from iree.turbine.kernel.boo.ops.graph import get_custom_graph_op
from .schema import FusionSchema
from .subgraph import (
    FusedSubgraph,
    extract_fusion_subgraph_modules,
)
from ....support.logging import aot_logger as logger

__all__ = [
    "fusion_transform",
]


def fusion_transform(module: fx.GraphModule, *, fusion_schema: FusionSchema) -> None:
    """Applies fusions to the underlying fx graph of a GraphModule by offloading subgraphs to IREE compiler/runtime."""

    _log_graph_module("Source module", module)

    subgraphs = extract_fusion_subgraph_modules(module, fusion_schema)
    subgraph_replacements: list[tuple[FusedSubgraph, fx.Node]] = []
    for subgraph in subgraphs:
        custom_op = get_custom_graph_op(subgraph.module, force_single_dispatch=False)
        # Insert call as early as possible, to maintain topological order.
        insert_pt = sorted(subgraph.arguments)[-1]
        with module.graph.inserting_after(insert_pt):
            call = module.graph.call_function(custom_op, tuple(subgraph.arguments))
        # Delay replacement until we've inserted all calls, so that the
        # references in 'subgraphs[i].arguments' are still valid.
        subgraph_replacements.append((subgraph, call))

    for subgraph, call in subgraph_replacements:
        _replace_with_call(module.graph, subgraph.results, call)
        for node in reversed(sorted(subgraph.matched_nodes)):
            module.graph.erase_node(node)

    module.recompile()
    module.graph.lint()

    _log_graph_module("Converted module", module)


def _replace_with_call(
    graph: fx.Graph,
    nodes_to_replace: Sequence[fx.Node],
    call: fx.Node,
):
    assert call.op == "call_function"
    with graph.inserting_after(call):
        outputs = (
            [call]
            if len(nodes_to_replace) == 1
            else [
                graph.call_method("__getitem__", args=(call, i))
                for i in range(len(nodes_to_replace))
            ]
        )
    for node_to_replace, call_output in zip(nodes_to_replace, outputs, strict=True):
        node_to_replace.replace_all_uses_with(call_output)


def _log_graph_module(label: str, gm: fx.GraphModule):
    logger.debug(
        "%s:\n%s", label, gm.print_readable(print_output=False, include_stride=True)
    )
