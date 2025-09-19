# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import math

from typing import Tuple, Iterable, NamedTuple, Sequence

import torch
from torch.fx.passes.shape_prop import TensorMetadata
from iree.compiler.extras.fx_importer import FxImporter

from ....support.logging import runtime_logger as logger
from ....support.ir_imports import Operation, PassManager, Context
from ....transforms.general.custom_op_expansion import ExpandCustomOpsPass

__all__ = [
    "is_boo_backward_enabled",
    "enable_backward",
    "disable_backward",
    "make_tuple",
    "CHANNELS_LAST_LAYOUTS",
    "CHANNELS_LAST_MEMORY_FORMAT",
    "CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION",
    "CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION",
    "get_arg_spec_name",
    "generate_custom_op_compatible_ir",
    "get_memory_format_permutation",
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


def _tensor_type_str(shape: Iterable[int], dtype: torch.dtype) -> str:
    dtype = str(dtype).removeprefix("torch.")
    shape_str = "x".join([str(dim) for dim in shape])
    return shape_str + f"x{dtype}"


class MemoryFormatPermutation(NamedTuple):
    """Contains a shape permutation used to convert a non-contiguous tensor into a contiguous format."""

    permutation: list[int]
    inverse_permutation: list[int]


def _try_get_permutations_from_strides(
    strides: Sequence[int], shape: Sequence[int]
) -> None | MemoryFormatPermutation:
    rank = len(strides)
    sorted_strides = sorted(
        list(zip(strides, range(rank), shape, strict=True)), reverse=True
    )

    # Check this metadata corresponds to a permutation of a contiguous tensor.
    numel = math.prod(shape)
    for stride, dim, size in sorted_strides:
        if size == 1:
            # Don't need to check anything here.
            continue
        if numel % size != 0:
            logger.debug(
                "Got size %s for numel %s at dim %s", str(size), str(numel), str(dim)
            )
            return None
        numel = numel // size
        if stride != numel:
            logger.debug(
                "Got stride %s for numel %s at dim %s", str(size), str(numel), str(dim)
            )
            return None

    # The middle items form a permutation which would result in a contiguous tensor.
    to_contig_perm = list([item[1] for item in sorted_strides])
    # Compute the inverse permutation.
    inverse_perm = [None] * rank
    for i in range(rank):
        inverse_perm[to_contig_perm[i]] = i
    return MemoryFormatPermutation(
        permutation=to_contig_perm, inverse_permutation=inverse_perm
    )


def get_memory_format_permutation(
    t: torch.Tensor,
    num_dims: int | None = None,
    *,
    strict: bool = False,
) -> MemoryFormatPermutation | None:
    """Returns a MemoryFormatPermutation for a Tensor if one can be inferred.
    This checks for `channels_last` and `channels_last_3d` memory_formats directly.

    If a tensor is neither in a `channels_last` format nor contiguous, then a
    MemoryFormatPermutation is directly inferred from the stride metadata.

    If a MemoryFormatPermutation cannot be inferred (e.g., t is a proper subview),
    then a warning is issued, or if `strict=True`, then an error is raised.
    """
    num_dims = num_dims or len(t.shape) - 2
    cl_mem_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_dims, None)
    if cl_mem_format is None or not t.is_contiguous(memory_format=cl_mem_format):
        if not t.is_contiguous(memory_format=torch.contiguous_format):
            stride = t.stride()
            maybe_perms = _try_get_permutations_from_strides(stride, t.shape)
            if maybe_perms is not None:
                return maybe_perms
            if strict:
                raise ValueError(
                    f"Expected tensor to be in contiguous or channels_last(_3d) memory formats. "
                    f"Got {type(t)} with {t.shape = }, {stride = }."
                )
            logger.warning(
                "Encountered tensor at kernel boundary with unhandled memory format. Got %s with shape %s, stride %s."
                "This tensor will be forced into contiguous format during dlpack handoff to IREE.",
                str(type(t)),
                str(t.shape),
                str(stride),
                exc_info=True,
            )
        return None

    cl_contig = CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION.get(num_dims)
    contig_cl = CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION.get(num_dims)
    return MemoryFormatPermutation(
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


def get_arg_spec_name_and_memory_format_permutations(
    base_name, *args
) -> tuple[str, list[MemoryFormatPermutation | None]]:
    name = base_name
    layout_handling: list[MemoryFormatPermutation | None] = []
    for idx, arg in enumerate(args):
        if arg is None:
            layout_handling.append(None)
            continue
        if not isinstance(arg, torch.Tensor):
            raise TypeError(
                f"Expected all function arguments to be (optional) tensors. Got {type(arg)} at position {idx}."
            )
        shape = arg.shape
        dtype = arg.dtype
        name += f"_{_tensor_type_str(shape, dtype)}"
        mem_format_info = get_memory_format_permutation(arg, len(shape) - 2)
        if mem_format_info:
            name += "_perm_" + "".join([str(i) for i in mem_format_info.permutation])
        layout_handling.append(mem_format_info)
    return name, layout_handling


def generate_custom_op_compatible_ir(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    func_name: str,
    context: Context,
) -> Operation:
    """Returns an mlir module operation.

    The provided torch.nn.Module is imported as an mlir function which can be inlined during ExpandCustomOpsPass.
    """
    importer = FxImporter(context=context)
    e = torch.export.export(module, args=args)
    importer.import_program(e, func_name=func_name, func_visibility="private")
    module_op = importer.module_op
    expansion_pass = ExpandCustomOpsPass(module_op)
    expansion_pass.run()
    pm = PassManager.parse(
        "builtin.module(canonicalize, torch-func-backend-type-conversion)",
        module_op.context,
    )
    pm.run(module_op)
    return module_op.operation
