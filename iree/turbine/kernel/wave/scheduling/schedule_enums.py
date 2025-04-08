# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum

"""
Formatting for different scheduling strategies:
Values: 0xAB where:
* A = Strategy types:
  * 0 = None
  * 1 = Solver Based
  * 2 = Heuristic Based
* B enumerates different strategy that share the same 0xA* bits.
"""


class SchedulingType(Enum):
    NONE = 0x00
    MODULO = 0x10
    PREFETCH = 0x20
    MODULO_MULTI_BUFFERED = 0x11
