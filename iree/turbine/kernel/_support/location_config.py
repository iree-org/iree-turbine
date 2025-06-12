# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum


class LocationCaptureLevel(Enum):
    NONE = 0
    FILE_LINE_COL = 1
    STACK_TRACE = 2
    STACK_TRACE_WITH_SYSTEM = 3


@dataclass
class LocationCaptureConfig:
    level: LocationCaptureLevel = LocationCaptureLevel.NONE
