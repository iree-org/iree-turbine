# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable
import torch
from ..runtime import get_launchable
from ..ops.utils import get_arg_spec_name

__all__ = [
    "kernel",
]


def kernel(fn: Callable) -> Callable:
    """Decorator to convert a dynamo exportable function into a launchable.

    This is not-yet compatible with the following:
        1. torch.compile (we aren't wrapping in a custom op).
        2. autograd (we aren't defining backward).
        3. automatic mixed precision.
        4. special handling of alternative memory formats like `torch.channels_last`.
    """

    class FakeMod(torch.nn.Module):
        def forward(self, *args):
            return fn(*args)

    def wrapped(*args):
        func_name = get_arg_spec_name(fn.__name__, *args)
        launch = get_launchable(
            lambda: FakeMod(),
            args,
            func_name=func_name,
            cache_only=False,
            force_single_dispatch=False,
        )
        return launch(*[arg.data for arg in args])

    return wrapped
