# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import functools

from typing import Tuple, Iterable

import torch

__all__ = [
    "is_boo_backward_enabled",
    "enable_backward",
    "disable_backward",
    "make_tuple",
    "CL_LAYOUT",
    "CL_MEM",
    "CL_CONTIGUOUS_PERM",
    "CONTIGUOUS_CL_PERM",
    "get_func_name",
]

# Toggle Using Boo Backward Kernels #

BOO_USE_BACKWARD_KERNELS = int(os.getenv("BOO_USE_BACKWARD_KERNELS", "0"))


def is_boo_backward_enabled():
    return bool(BOO_USE_BACKWARD_KERNELS)


def enable_backward():
    """Allows toggling on Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 1


def disable_backward():
    """Allows toggling off Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 0


# Utilities #


def make_tuple(a: Iterable | int, size: int) -> Tuple:
    """Tries to convert `a` into a Tuple of ints."""
    if isinstance(a, Iterable):
        result = tuple(a)
        assert len(result) == size
        assert isinstance(result[0], int)
        return result
    if isinstance(a, int):
        return (a,) * size
    raise TypeError(f"Input {a} is expected to be an iterable or int. Got {type(a)}.")


CL_LAYOUT = {1: "NHC", 2: "NHWC", 3: "NDHWC"}
CL_MEM = {2: torch.channels_last, 3: torch.channels_last_3d}
CL_CONTIGUOUS_PERM = {1: [0, 2, 1], 2: [0, 2, 3, 1], 3: [0, 2, 3, 4, 1]}
CONTIGUOUS_CL_PERM = {1: [0, 2, 1], 2: [0, 3, 1, 2], 3: [0, 4, 1, 2, 3]}


@functools.lru_cache(maxsize=None)
def get_func_name(
    input_shape: tuple,
    kernel_shape: tuple,
    dtype: str,
    mode: str,
    bias: bool,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
) -> str:
    num_spatial_dims = len(input_shape) - 2
    name_items = [
        "conv",
        f"{num_spatial_dims}d",
        str(dtype).removeprefix("torch."),
        str(mode).lower(),
    ]
    if bias and mode == "FORWARD":
        name_items.append("b")
    l2s = lambda l: "x".join([str(i) for i in l])
    name_items.extend(
        [
            l2s(input_shape),
            input_layout.lower(),
            l2s(kernel_shape),
            kernel_layout.lower().replace("n", "f"),
            output_layout.lower().replace("c", "f"),
            l2s(stride) + "s",
            l2s(padding) + "p",
            l2s(dilation) + "d",
            f"{groups}g",
        ]
    )
    return "_".join(name_items)
