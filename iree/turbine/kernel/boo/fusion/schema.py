# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from torch.fx.node import Target

from .replacement import ReplacementSchema, replace_aten_convolution


@dataclass
class OpFusionSpec:
    recursive: bool = True
    producers: Sequence[Target] = ()
    consumers: Sequence[Target] = ()


FusionSchema = Dict[Target, OpFusionSpec]

# TODO: extend this
DEFAULT_SUPPORTED_BOO_FUSIONS: FusionSchema = {
    torch.ops.aten.convolution.default: OpFusionSpec()
}

DEFAULT_POST_FUSION_REPLACEMENTS: ReplacementSchema = {
    torch.ops.aten.convolution.default: replace_aten_convolution
}
