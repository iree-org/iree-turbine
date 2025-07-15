# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from typing import Any, Union, Tuple


def latency_to_us(latency: float) -> float:
    """Convert latency from seconds to microseconds."""
    return latency * 1_000_000 if latency != float("inf") else float("inf")


def format_latency_us(latency: float, precision: int = 2) -> str:
    """Format latency in microseconds with specified precision."""
    if latency == float("inf"):
        return "inf"
    return f"{latency_to_us(latency):.{precision}f} μs"


def enum_to_str(obj: Union[Enum, Tuple, Any]) -> Any:
    """Convert enum objects to strings for JSON serialization."""
    if isinstance(obj, Enum):
        return str(obj)
    elif isinstance(obj, tuple):
        return tuple(enum_to_str(x) for x in obj)
    else:
        return obj
