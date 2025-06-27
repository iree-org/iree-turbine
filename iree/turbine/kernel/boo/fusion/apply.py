# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

import torch
from torch.fx.graph_module import GraphModule
from torch._functorch.aot_autograd import _aot_export_function
from .schema import FusionSchema, DEFAULT_SUPPORTED_BOO_FUSIONS
from .subgraph import extract_fusion_subgraph_modules, fused_subgraph, replace_subgraphs
from ..ops.graph import get_autograd_function
from ....support.logging import aot_logger as logger

__all__ = [
    "fusion_transform",
]


def _log_graph_module(label: str, gm: GraphModule):
    logger.debug("%s:\n%s", label, gm.print_readable(print_output=False))


def fusion_transform(
    module: torch.nn.Module,
    args: Tuple[torch.Tensor],
    *,
    fusion_schema: FusionSchema = DEFAULT_SUPPORTED_BOO_FUSIONS,
) -> torch.nn.Module:
    """Applies fusions to the underlying fx graph for module by offloading subgraphs to IREE compiler/runtime.

    This function expects the model to contain exclusively tensor arguments.

    This currently uses dynamo to export a graph, from which we auto-generate custom boo ops to replace fusable subgraphs.
    """

    if not all([isinstance(a, torch.Tensor) for a in args]):
        raise ValueError("fusion_transform expects model arguments to be tensors.")

    exported_program = torch.export.export(module, args=args)

    gm: GraphModule = exported_program.graph_module

    _log_graph_module("Source Graph Module", gm)

    subgraphs, _ = extract_fusion_subgraph_modules(gm, fusion_schema)
    subgraph_repl = []
    for sg in subgraphs:
        sg.print_readable()
        _log_graph_module("Extracted SubGraph Module", sg)
        fake_args = tuple(
            [n.meta.get("val") for n in sg.graph.nodes if n.op == "placeholder"]
        )
        joint_sg, metadata, in_spec, out_spec = _aot_export_function(
            sg.forward,
            fake_args,
            decompositions=None,
        )
        # TODO: do some minimal validation on the results of the above function.
        # in_spec, _kw_in_spec = in_spec.children_specs
        _log_graph_module("AOT Joint FWD/BWD Subgraph Module", joint_sg)
        joint_sg.print_readable()
        fake_args_joint = tuple(
            [n.meta.get("val") for n in joint_sg.graph.nodes if n.op == "placeholder"]
        )

        custom_op = get_autograd_function(
            joint_sg, fake_args_joint, num_fwd_outputs=metadata.num_forward_returns
        )

        single_node_graph = fused_subgraph(
            sg,
            custom_op,
            (n for n in sg.graph.nodes if n.op == "placeholder"),
            num_outputs=metadata.num_forward_returns,
        )
        _log_graph_module("Replacement Subgraph", joint_sg)
        subgraph_repl.append(single_node_graph)

    _ = replace_subgraphs(gm, subgraphs, subgraph_repl)

    # TODO: update any metadata which may have been modified by the replacement.

    logger.debug("Converted exported program:\n%s", str(exported_program))

    converted_module = exported_program.module()

    logger.debug("Converted module:\n%s", str(converted_module))

    return converted_module
