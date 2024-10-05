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
    from ..lang.types import Index

from .base import (
    define_op,
)

__all__ = [
    "for_loop",
]


@define_op
def for_loop(
    start: "Index",
    stop: Optional["Index"] = None,
    step: Optional["Index"] = None,
    init_args: List[Any] = [],
) -> Callable[[Callable[["Index", List[Any]], Optional[Tuple]]], List[Any]]:
    # TODO: The output type signature should also allow a single element return
    # instead of a List for better programming experience.
    ...
