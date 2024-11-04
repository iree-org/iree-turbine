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


def infer_types(trace: CapturedTrace | fx.Graph):
    # Infer and set the types for all nodes in the graph.
    for subgraph in trace.region_graph.subgraphs.values():
        for node in subgraph.nodes:
            custom = get_custom(node)
            custom.infer_type()
            logger.debug(f"Setting type for {custom.fx_node} = {custom.type}")
