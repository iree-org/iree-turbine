# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import functools

from typing import Tuple, Iterable, NamedTuple

import torch
from torch.fx.passes.shape_prop import TensorMetadata

__all__ = [
    "is_boo_backward_enabled",
    "enable_backward",
    "disable_backward",
    "make_tuple",
    "CHANNELS_LAST_LAYOUTS",
    "CHANNELS_LAST_MEMORY_FORMAT",
    "CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION",
    "CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION",
    "get_func_name",
    "get_arg_spec_name",
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


CHANNELS_LAST_LAYOUTS = {1: "NHC", 2: "NHWC", 3: "NDHWC"}
CHANNELS_LAST_MEMORY_FORMAT = {2: torch.channels_last, 3: torch.channels_last_3d}
CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION = {
    1: [0, 2, 1],
    2: [0, 2, 3, 1],
    3: [0, 2, 3, 4, 1],
}
CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION = {
    1: [0, 2, 1],
    2: [0, 3, 1, 2],
    3: [0, 4, 1, 2, 3],
}


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
    to_shape_string = lambda l: "x".join([str(i) for i in l])
    name_items.extend(
        [
            to_shape_string(input_shape),
            input_layout.lower(),
            to_shape_string(kernel_shape),
            kernel_layout.lower().replace("n", "f"),
            output_layout.lower().replace("c", "f"),
            to_shape_string(stride) + "s",
            to_shape_string(padding) + "p",
            to_shape_string(dilation) + "d",
            f"{groups}g",
        ]
    )
    return "_".join(name_items)


def _tensor_type_str(shape: Iterable[int], dtype: torch.dtype) -> str:
    dtype = str(dtype).removeprefix("torch.")
    shape_str = "x".join([str(dim) for dim in shape])
    return shape_str + f"x{dtype}"


def _is_contiguous(
    t: torch.Tensor | TensorMetadata, memory_format: torch.memory_format
) -> bool:
    if isinstance(t, torch.Tensor):
        return t.is_contiguous(memory_format=memory_format)
    metadata_memory_format = getattr(t, "memory_format", None)
    if metadata_memory_format is None:
        if isinstance(t, TensorMetadata):
            raise ValueError(
                f"TensorMetadata: {t} does not have memory_format attribute!"
            )
        raise TypeError(
            f"Unhandled type: {type(t)}. _is_contiguous input 0 must be a torch.Tensor or TensorMetadata object."
        )


class MemoryFormatInformation(NamedTuple):
    is_channels_last: bool = False
    # Whether the tensor is in channels_last format
    permutation: None | list[int] = None
    # A permutation which will materialize the channels_last format into the tensor's shape.
    inverse_permutation: None | list[int] = None
    # The inverse of permutation.


def get_memory_format_information(
    t: torch.Tensor | TensorMetadata, num_dims: int | None = None
) -> MemoryFormatInformation:
    """For a torch.Tensor, returns a tuple contatining:

    1. A bool indicating whether the tensor is in a num-dim-appropriate channels-last format.
    2. An optional list indicating the channels-last to contiguous layout permutation.
    3. An optional list indicating the inverse permutation.
    """
    num_dims = num_dims or len(t.shape) - 2
    cl_mem_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_dims, None)
    if cl_mem_format is None or not _is_contiguous(t, cl_mem_format):
        return MemoryFormatInformation()

    cl_contig = CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION.get(num_dims)
    contig_cl = CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION.get(num_dims)
    if None in [cl_contig, contig_cl]:
        return MemoryFormatInformation()
    return MemoryFormatInformation(
        is_channels_last=True,
        permutation=cl_contig,
        inverse_permutation=contig_cl,
    )


def get_arg_spec_name(base_name, *args):
    name = base_name
    for idx, arg in enumerate(args):
        if arg is not None and not isinstance(arg, torch.Tensor):
            raise TypeError(
                f"Expected all function arguments to be (optional) tensors. Got {type(arg)} at position {idx}."
            )
        name += f"_{_tensor_type_str(arg.shape, arg.dtype)}"
    return name


def get_arg_spec_name_and_memory_format_information(
    base_name, *args
) -> tuple[str, list[MemoryFormatInformation]]:
    name = base_name
    layout_handling: list[MemoryFormatInformation] = []
    for idx, arg in enumerate(args):
        if arg is None:
            layout_handling.append((False, None, None))
            continue
        if not isinstance(arg, torch.Tensor):
            raise TypeError(
                f"Expected all function arguments to be (optional) tensors. Got {type(arg)} at position {idx}."
            )
        shape = arg.shape
        dtype = arg.dtype
        name += f"_{_tensor_type_str(shape, dtype)}"
        mem_format_info = get_memory_format_information(arg, len(shape) - 2)
        if mem_format_info[0]:
            name += "_cl"
        layout_handling.append(mem_format_info)
    return name, layout_handling
