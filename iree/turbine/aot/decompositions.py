# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
from typing import Optional

import torch

from ..dynamo.decompositions import (
    _current,
    _extend_context_manager,
    DecompositionOpsList,
    DecompositionTable,
)

__all__ = [
    "current_aot_decompositions",
    "extend_aot_decompositions",
]


def current_aot_decompositions() -> DecompositionTable:
    """Gets the current decomposition table for AOT."""
    return _current("aot")


def extend_aot_decompositions(
    *,
    from_current: bool = True,
    add_ops: Optional[DecompositionOpsList] = None,
    remove_ops: Optional[DecompositionOpsList] = None
):
    """Context manager which extends the list of decompositions used for AOT."""
    return _extend_context_manager(
        "aot", from_current=from_current, add_ops=add_ops, remove_ops=remove_ops
    )
