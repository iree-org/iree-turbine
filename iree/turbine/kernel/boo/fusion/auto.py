# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable
import torch
from ..runtime import get_launchable

__all__ = [
    "kernel",
]


def tensor_type_str(t: torch.Tensor | None) -> str:
    if t is None:
        return ""
    shape = t.shape
    dtype = str(t.dtype).removeprefix("torch.")
    shape_str = "x".join([str(dim) for dim in shape])
    return shape_str + f"x{dtype}"


def get_name(f, *args):
    name = f.__name__
    for idx, arg in enumerate(args):
        if arg is not None and not isinstance(arg, torch.Tensor):
            raise TypeError(
                f"Expected all function arguments to be (optional) tensors. Got {type(arg)} at position {idx}."
            )
        name += f"_{tensor_type_str(arg)}"
    return name


def kernel(fn: Callable | torch.nn.Module) -> Callable:
    """Decorator to convert a dynamo exportable function into a launchable."""

    class FakeMod(torch.nn.Module):
        def forward(self, *args):
            return fn(*args)

    def wrapped(*args):
        func_name = get_name(fn, *args)
        launch = get_launchable(
            lambda: FakeMod(),
            args,
            func_name=func_name,
            cache_only=False,
            force_single_dispatch=False,
        )
        return launch(*[arg.data for arg in args])

    return wrapped
