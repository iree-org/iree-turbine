# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..._support.indexing import IndexExpr, IndexingContext, IndexSymbol, IndexSequence
from typing import Any
import sympy


def safe_subs(
    input: IndexExpr | int,
    subs: dict[IndexSymbol, int | IndexSymbol],
    simultaneous: bool = False,
) -> IndexSymbol | int:
    """
    Substitute input using provided `subs` list if input is sympy object.
    Otherwise return input unchanged.
    """
    if isinstance(input, (sympy.Basic, IndexSequence)):
        return input.subs(subs, simultaneous=simultaneous)  # type: ignore

    return input


def subs_idxc(input: Any) -> IndexSymbol | int:
    """
    Substitute input using IndexingContext if input is sympy object.
    Otherwise return input unchanged.
    """
    idxc = IndexingContext.current()
    return safe_subs(input, idxc.subs)
