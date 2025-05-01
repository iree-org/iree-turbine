# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ops.wave_ops import *
from .._support.tracing import CapturedTrace
import torch.fx as fx
from ...support.logging import get_logger

logger = get_logger("turbine.wave.type_inference")


def infer_types(trace: CapturedTrace, subgraph: Optional[fx.Graph] = None):
    if subgraph:
        all_nodes = subgraph.nodes
    else:
        all_nodes = trace.get_root_graph().nodes
    # Infer and set the types for all nodes in the graph.
    for node in all_nodes:
        custom = get_custom(node)
        if isinstance(custom, NestedRegionOp):
            infer_types(trace, trace.region_graph.subgraphs[custom.subgraph_name])
        custom.infer_type()
        # For implicit captures, get type from variables in root graph.
        if "lifted" in custom.fx_node.meta:
            custom.type = custom.fx_node.meta["lifted"].type
        logger.debug(f"Setting type for {custom.fx_node} = {custom.type}")
