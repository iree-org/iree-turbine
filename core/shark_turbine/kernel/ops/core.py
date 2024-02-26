from typing import Any, TypeVar
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Index, Vector

from .base import define_op
from .._support.dtype import DataType

__all__ = [
    "kernel_buffer_getitem",
    "kernel_buffer_setitem",
    "thread_program_id",
    "to_dtype",
]


@define_op
def kernel_buffer_getitem(kernel_buffer, key) -> "Vector":
    ...


@define_op
def kernel_buffer_setitem(kernel_buffer, key, item) -> None:
    ...


@define_op
def thread_program_id(axis: int) -> "Index":
    ...


@define_op
def to_dtype(val, dtype: DataType):
    ...
