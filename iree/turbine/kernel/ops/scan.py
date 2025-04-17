import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_cumsum",
]


@define_op
def vector_cumsum(vector: "Vector", axis=None, acc=None) -> "Vector":
    ...
