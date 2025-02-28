# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.support.tools import iree_tool_prepare_input_args
from pathlib import Path


def test_iree_tool_prepare_input_args(tmp_path: Path):
    arg0 = torch.tensor([1.1, 2.2, 3.3, 4.4], dtype=torch.bfloat16)
    arg1 = torch.tensor([[4, 5], [6, 7]], dtype=torch.int8)
    args = [arg0, arg1]
    cli_arg_values = iree_tool_prepare_input_args(
        args, file_path_prefix=tmp_path / "arg"
    )

    expected_arg_file_paths = [
        tmp_path / "arg0.bin",
        tmp_path / "arg1.bin",
    ]

    assert cli_arg_values[0] == f"4xbf16=@{expected_arg_file_paths[0]}"
    assert cli_arg_values[1] == f"2x2xi8=@{expected_arg_file_paths[1]}"

    for arg, file_path in zip(args, expected_arg_file_paths):
        with open(file_path, "rb") as f:
            actual_bytes = f.read()
            assert arg.cpu().view(dtype=torch.int8).numpy().tobytes() == actual_bytes
