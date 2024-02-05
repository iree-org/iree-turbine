from typing import Any, List, Optional
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Vector

from .base import (
    define_op,
)

__all__ = [
    "vector_max",
    "vector_sum",
    "vector_dot",
]


@define_op
def vector_max(vector: "Vector", axis=None, acc=None) -> "Vector":
    ...


@define_op
def vector_sum(vector: "Vector", axis=None, acc=None) -> "Vector":
    ...


@define_op
def vector_dot(lhs: "Vector", rhs: "Vector", acc=None) -> "Vector":
    ...
