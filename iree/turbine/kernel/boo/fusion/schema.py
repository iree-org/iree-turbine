# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Dict, Sequence
from torch.fx.node import Target

# import torch


@dataclass
class OpFusionSpec:
    recursive: bool = True
    producers: Sequence[Target] = ()
    consumers: Sequence[Target] = ()


FusionSchema = Dict[Target, OpFusionSpec]
