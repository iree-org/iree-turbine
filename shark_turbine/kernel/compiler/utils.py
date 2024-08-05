from typing import Optional
from .._support.indexing import IndexSymbol, IndexingContext
from math import prod


def strides_from_symbolic_shape(
    indexing_context: IndexingContext,
    symbolic_shape: Optional[list[IndexSymbol]],
) -> Optional[list[int]]:
    """
    Computes the stride from a given symbolic shape and indexing context,
    assuming the innermost dimension is contiguous.
    """
    if symbolic_shape is None:
        return None
    static_shape = [indexing_context.get_static_value(sym) for sym in symbolic_shape]
    if None in static_shape:
        return None
    strides = []
    for i in range(1, len(static_shape)):
        strides.append(prod(static_shape[-i:]))
    return strides[::-1] + [1]
