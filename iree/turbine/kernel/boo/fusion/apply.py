# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple

import torch
from torch.fx.graph_module import GraphModule
from .schema import FusionSchema, DEFAULT_SUPPORTED_BOO_FUSIONS
from .subgraph import (
    extract_fusion_subgraph_modules,
    replace_subgraphs,
    get_subgraph_replacement,
    _log_graph_module,
)
from ....support.logging import aot_logger as logger

__all__ = [
    "fusion_transform",
]


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
        subgraph_repl.append(get_subgraph_replacement(sg))

    _ = replace_subgraphs(gm, subgraphs, subgraph_repl)

    # TODO: update any metadata which may have been modified by the replacement.

    logger.debug("Converted exported program:\n%s", str(exported_program))

    converted_module = exported_program.module()

    logger.debug("Converted module:\n%s", str(converted_module))

    return converted_module
