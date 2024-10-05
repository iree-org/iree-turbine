from typing import (
    Any,
    List,
    Tuple,
    Optional,
    Iterator,
    overload,
    Callable,
    Tuple,
)
import typing

if typing.TYPE_CHECKING:
    from ..lang.types import Index, Vector

from .base import (
    define_op,
)

__all__ = ["kernel_buffer_load", "kernel_buffer_store"]


@define_op
def kernel_buffer_load(
    kernel_buffer,
    multi_index: Tuple["Index", ...],
    shape: Tuple[int, ...],
) -> "Vector":
    ...


@define_op
def kernel_buffer_store(
    kernel_buffer,
    multi_index: Tuple["Index", ...],
    item: "Vector",
) -> None:
    ...
