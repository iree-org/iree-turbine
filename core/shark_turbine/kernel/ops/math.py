from typing import Tuple
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "exp2",
    "vector_constant",
]


@define_op
def exp2(val):
    ...


@define_op
def vector_constant(shape: Tuple[int, ...], dtype, value: int | float) -> "Vector":
    ...
