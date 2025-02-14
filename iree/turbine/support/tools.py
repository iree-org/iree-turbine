# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Collection, Iterable
from os import PathLike

from .conversions import torch_dtyped_shape_to_iree_format


def iree_tool_format_cli_input_arg(arg: torch.Tensor, file_path: str | PathLike) -> str:
    """Format the CLI value for an input argument.
    Example:

    .. code-block:: Python

        iree_tool_format_cli_input_arg(torch.empty([1,2], dtype=torch.float32), "arg0.bin")
    Returns:

    .. code-block:: Python

        "1x2xf32=@arg0.bin"
    """
    return f"{torch_dtyped_shape_to_iree_format(arg)}=@{file_path}"


def write_raw_tensor(tensor: torch.Tensor, file_path: str | PathLike):
    """Write the contents of the tensor as they are in memory without any metadata."""
    with open(file_path, "wb") as f:
        f.write(tensor.cpu().view(dtype=torch.int8).numpy().data)


def iree_tool_prepare_input_args(
    args: Collection[torch.Tensor],
    /,
    *,
    file_paths: Iterable[str | PathLike] | None = None,
    file_path_prefix: str | PathLike | None = None,
) -> list[str]:
    """Write the raw contents of tensors to files without any metadata.
    Returns the CLI input args description.

    If :code:`file_path_prefix` is given, will chose a default naming for argument
    files. It is treated as a string prefix and not as directory.
    Example:

    .. code-block:: Python

        iree_tool_prepare_input_args(
            args,
            file_path_prefix="/some/path/arg",
        )


    returns

    .. code-block:: Python

        [
            "1x2x3xf32=@/some/path/arg0.bin",
            "4x5xi8=@/some/path/arg1.bin"
        ]

    This results can be prefixed with :code:`--input=` to arrive at the final CLI flags
    expected by IREE tools.

    Exactly one of file_paths and file_path_prefix must be provided.
    Does not create parent directory(s).
    """
    if file_paths is not None and file_path_prefix is not None:
        raise ValueError(
            "file_paths and file_path_prefix are mutually exclusive arguments."
        )
    if file_paths is None and file_path_prefix is None:
        raise ValueError("One of file_paths and file_path_prefix must be provided.")

    if file_paths is None:
        file_paths = [f"{file_path_prefix}{i}.bin" for i in range(len(args))]
    for tensor, file_path in zip(args, file_paths, strict=True):
        write_raw_tensor(tensor, file_path)
    return [
        iree_tool_format_cli_input_arg(tensor, file_path)
        for tensor, file_path in zip(args, file_paths, strict=True)
    ]
