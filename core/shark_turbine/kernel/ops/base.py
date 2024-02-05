"""Support for defining the op library and dispatch."""

from typing import Callable, Type, TypeVar

import functools

from .._support import context

T = TypeVar("T")


class OpDispatcher:
    """Handles dispatch of operations by their idname.

    Operations are dispatched by looking up a function on the dispatcher like:
      def handle_{idname}(self, operator, *args, **kwargs)
    """

    __tk_context_idname__ = "OpDispatcher"

    @staticmethod
    def current() -> "OpDispatcher":
        return context.current(OpDispatcher)

    def __enter__(self) -> "OpDispatcher":
        return context.push(OpDispatcher, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(OpDispatcher, self)


def define_op(f: T) -> T:
    idname = f.__name__

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        dispatcher = OpDispatcher.current()
        try:
            handler = getattr(dispatcher, f"handle_{idname}")
        except AttributeError:
            raise AttributeError(
                f"The current OpDispatcher ({dispatcher}) does not register a handler for {idname}"
            )
        return handler(wrapped, *args, **kwargs)

    wrapped.__tk_op_idname__ = idname
    return wrapped
