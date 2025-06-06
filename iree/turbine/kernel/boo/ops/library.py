# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from typing import Callable

__all__ = [
    "define_schema",
    "register_impl",
    "register_meta",
]

BOO_LIBRARY = torch.library.Library("boo", "DEF")


# Library Helpers #


def define_schema(name: str, schema: str):
    BOO_LIBRARY.define(f"{name}{schema}")


def register_impl(name: str):
    """Registers the wrapped function as an implementation for CPU and CUDA for op with specified name."""

    def wrap(fn: Callable):
        for key in ["CPU", "CUDA"]:
            BOO_LIBRARY.impl(name, fn, key)
        return fn

    return wrap


def register_meta(name: str):
    """Registers the wrapped function as a meta implementation for op with specified name."""

    def wrapper(fn: Callable):
        BOO_LIBRARY.impl(name, fn, "Meta")
        return fn

    return wrapper
