# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Sequence

import torch
from torch import fx
from torch.fx.experimental.proxy_tensor import make_fx
from operator import getitem

from iree.turbine.kernel.boo.ops.graph import get_custom_graph_op
from .schema import FusionSchema, ReplacementSchema
from .replacement import apply_replacements
from .subgraph import (
    FusedSubgraph,
    extract_fusion_subgraph_modules,
)
from ....support.logging import aot_logger as logger
from ....dynamo.decompositions import DEFAULT_DECOMPOSITION_TABLE

__all__ = [
    "fusion_transform",
]


def infer_example_inputs(
    graph_module: fx.GraphModule,
) -> tuple[torch.Tensor, ...]:
    fake_inputs = tuple(
        n.meta["val"] for n in graph_module.graph.find_nodes(op="placeholder")
    )
    assert all(
        [isinstance(fake_inp, torch.Tensor) for fake_inp in fake_inputs]
    ), f"Expected all placeholder `meta['val']` to be tensors. Got {fake_inputs}."
    return fake_inputs


def fusion_transform(
    module: fx.GraphModule,
    *,
    fusion_schema: FusionSchema,
    post_fusion_replacements: ReplacementSchema,
    post_decomposition_replacements: ReplacementSchema,
) -> None:
    """Applies fusions to the underlying fx graph of a GraphModule by offloading subgraphs to IREE compiler/runtime."""

    _log_graph_module("Source module", module)

    subgraphs = extract_fusion_subgraph_modules(module, fusion_schema)

    # Replace subgraphs with custom (templated MLIR) kernels if required.
    subgraph_replacements: list[tuple[FusedSubgraph, fx.Node]] = []
    for subgraph in subgraphs:
        # Unfortunately, we need to do two stages of replacement/decomposition.
        # Firstly, direct graph modifications are done for specific ops.
        apply_replacements(subgraph.module.graph, post_fusion_replacements)
        subgraph.module.recompile()
        # Secondly, we apply some default fx decompositions.
        # This will re-trace the graph in a fake execution context.
        # An added benefit is the canonicalization of each subgraph, which
        # reduces the number of redundant custom ops being generated.
        decomposed_gm = make_fx(
            subgraph.module,
            decomposition_table=DEFAULT_DECOMPOSITION_TABLE,
            tracing_mode="fake",
        )(*infer_example_inputs(subgraph.module))
        _log_graph_module("Decomposed module", decomposed_gm)

        # For ops that require replacements after decomposition.
        apply_replacements(decomposed_gm.graph, post_decomposition_replacements)
        decomposed_gm.graph.eliminate_dead_code()
        decomposed_gm.recompile()

        custom_op = get_custom_graph_op(
            decomposed_gm, force_single_dispatch=subgraph.single_dispatch
        )
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

    _log_graph_module("Post-fusion module", module)


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
                graph.call_function(getitem, args=(call, i))
                for i in range(len(nodes_to_replace))
            ]
        )
    for node_to_replace, call_output in zip(nodes_to_replace, outputs, strict=True):
        node_to_replace.replace_all_uses_with(call_output, propagate_meta=True)


def _log_graph_module(label: str, gm: fx.GraphModule):
    logger.debug(
        "%s:\n%s", label, gm.print_readable(print_output=False, include_stride=True)
    )
