# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from .constraints import (
    Constraint,
    GenericDot,
)
import torch.fx as fx
from ..ops.wave_ops import get_custom


def _is_dot_mma(node: fx.Node) -> bool:
    return isinstance(get_custom(node), GenericDot)


def decompose_dot_mma(trace: CapturedTrace, constraints: list[Constraint]):
    print("Decomposing dot mma")
    pass
