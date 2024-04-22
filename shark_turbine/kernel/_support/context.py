from typing import Optional, Type, TypeVar

import threading

_tls = threading.local()

T = TypeVar("T")


def push(context_type: Type[T], instance: T) -> T:
    """Pushes an instance onto a thread-local context stack.

    The context type must define an attribute __tk_context_idname__ which is
    a valid/unique identifier.
    """
    assert isinstance(instance, context_type)
    key = context_type.__tk_context_idname__
    try:
        stack: list = getattr(_tls, key)
    except AttributeError:
        stack = []
        setattr(_tls, key, stack)
    stack.append(instance)
    return instance


def pop(context_type: Type[T], expected: Optional[T] = None):
    """Pops the current context off of the stack.

    Raises IndexError if no current.
    """
    stack: list = getattr(_tls, context_type.__tk_context_idname__)
    instance = stack.pop()
    assert (
        expected is None or expected is instance
    ), f"mismatched context push/pop for {context_type}"


def current(context_type: Type[T]) -> T:
    """Returns the current context from the stack.

    Raises IndexError on failure.
    """
    try:
        stack: list = getattr(_tls, context_type.__tk_context_idname__)
    except AttributeError:
        raise IndexError(f"No current context for {context_type}")
    try:
        instance = stack[-1]
    except IndexError:
        raise IndexError(f"No current context for {context_type}")
    assert isinstance(instance, context_type)
    return instance
