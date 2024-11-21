from typing import Optional
from .._support.indexing import IndexSymbol, IndexingContext
from math import prod


def strides_from_symbolic_shape(
    indexing_context: IndexingContext,
    symbolic_shape: Optional[list[IndexSymbol]],
    allow_mixed_shapes: bool = False,
) -> Optional[list[int]]:
    """
    Computes the stride from a given symbolic shape and indexing context,
    assuming the innermost dimension is contiguous.
    """
    if symbolic_shape is None:
        return None
    static_shape = [indexing_context.get_static_value(sym) for sym in symbolic_shape]
    if None in static_shape and not allow_mixed_shapes:
        return None
    mixed_shape = [
        static if static is not None else dynamic
        for static, dynamic in zip(static_shape, symbolic_shape)
    ]
    strides = []
    for i in range(1, len(mixed_shape)):
        strides.append(prod(mixed_shape[-i:]))
    return strides[::-1] + [1]
