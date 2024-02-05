from typing import Optional

import torch
from torch import Tensor

import functools
import inspect


def signature_matcher(f=None, *, arity: Optional[int] = None, original_name: str = ""):
    """Transforms a function into a signature matcher.

    The transfored function takes the same args/kwargs as the original, but
    it will return an inspect.BoundArguments.arguments when invoked.

    Optional overload selectors can be specified, and if not met, None
    will be returned (versus raising an error).

    On argument mismatch, a TypeError will be raised.
    """
    if f is None:
        return functools.partial(
            signature_matcher, arity=arity, original_name=original_name
        )

    sig = inspect.signature(f)

    def wrapped(*args, **kwargs) -> Optional[inspect.BoundArguments]:
        if arity is not None and arity != (len(args) + len(kwargs)):
            return None
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            return bound_args.arguments
        except TypeError as e:
            reported_name = original_name or f.__name__
            raise TypeError(f"{reported_name}() {str(e)}")

    return wrapped


@signature_matcher(original_name="torch.exp")
def torch_exp(input: Tensor) -> Tensor:
    ...


@signature_matcher(arity=1, original_name="torch.max")
def torch_max_unary(input: Tensor) -> Tensor:
    ...


@signature_matcher(original_name="torch.max")
def torch_max(input: Tensor, dim: int, keepdim: bool = False):
    ...


@signature_matcher(arity=1, original_name="torch.sum")
def torch_sum_unary(input: Tensor) -> Tensor:
    ...


@signature_matcher(original_name="torch.sum")
def torch_sum(input: Tensor, dim: int, keepdim: bool = False):
    ...
