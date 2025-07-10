# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from dataclasses import dataclass


@dataclass
class KernelLaunchInfo:
    grid: tuple[int] = None
    blocks: tuple[int] = None
    shared_memory_bytes: int = 0
    func_name: str = ""
    grid_str: str = ""


############################################
# Wave Ops related Utils
############################################


# GPU shuffle modes
class ShuffleMode(Enum):
    XOR = 0
    DOWN = 1
    UP = 2
    IDX = 3

class LDSTransposeRead(Enum):
    tr8_b64 = 0