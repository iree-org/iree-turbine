# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..._support.indexing import IndexExpr, IndexingContext, IndexSymbol, IndexSequence
from typing import Any, List, Tuple
import sympy


def safe_subs(input: Any, subs: List[Tuple[IndexExpr, IndexExpr]]) -> Any:
    """
    Substitute input using provided `subs` list if input is sympy object.
    Otherwise return input unchanged.
    """
    if isinstance(input, (sympy.Basic, IndexSequence)):
        return input.subs(subs)

    return input


def subs_idxc(input: Any) -> Any:
    """
    Substitute input using IndexingContext if input is sympy object.
    Otherwise return input unchanged.
    """
    idxc = IndexingContext.current()
    return safe_subs(input, idxc.subs)


def infer_static_shape(
    shape: tuple[int | IndexExpr], idxc: IndexingContext
) -> tuple[int, ...]:
    """Infer the static shape using the indexing context."""
    shape = tuple(idxc.get_static_value(safe_subs(sym, idxc.subs)) for sym in shape)
    if None in shape:
        raise ValueError(f"A dynamic dim found in the shape")
    return shape


def delinearize_index(
    index: IndexExpr, shape: list[int | IndexExpr]
) -> list[IndexExpr]:
    """
    Delinearizes a 1D index into a multi-dimensional index
    based on the shapes provided. The returned array contains
    the multi-dimensional index.

    Assume the index is x and the shape is [5, 4, 3]. In this case,
    this function returns [x % 3, (x // 3) % 4, (x // 12) % 5].

    """
    nd_index = []
    product = 1
    for i, size in enumerate(reversed(shape)):
        if i == 0:
            nd_index.append(index % size)
        else:
            nd_index.append(sympy.floor(index / product) % size)
        product *= size
    return nd_index[::-1]
