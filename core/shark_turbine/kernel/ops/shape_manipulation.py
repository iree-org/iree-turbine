from typing import Tuple
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_broadcast",
    "vector_broadcast_in_dim",
    "vector_transpose",
]


@define_op
def vector_broadcast(v: "Vector", leading_sizes: Tuple[int]) -> "Vector":
    ...


@define_op
def vector_broadcast_in_dim(
    v: "Vector", shape: Tuple[int], broadcast_dimensions: Tuple[int]
) -> "Vector":
    ...


@define_op
def vector_transpose(v: "Vector", permutation: Tuple[int]) -> "Vector":
    ...
