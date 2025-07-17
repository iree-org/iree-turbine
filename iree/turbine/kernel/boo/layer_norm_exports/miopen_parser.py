# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch

from iree.turbine.kernel.boo.layer_norm_exports.layer_norm import (
    LayerNormSignature,
    Mode,
)
from iree.turbine.kernel.boo.exports.parser import OpCLIParser


def _parse_shape(shape: str) -> list[int]:
    for symbol in shape:
        assert symbol in "0123456789x", "Unsupported shape syntax."

    return list(map(int, shape.split("x")))


class _DTypeCommandDispatcher:
    SUPPORTED = {
        "layernorm": torch.float,
        "layernormfp16": torch.float16,
        "layernormbfp16": torch.bfloat16,
    }

    @staticmethod
    def choices() -> list[str]:
        return list(_DTypeCommandDispatcher.SUPPORTED.keys())

    @staticmethod
    def get_dtype(command: str) -> torch.dtype:
        assert (
            command in _DTypeCommandDispatcher.SUPPORTED
        ), f"Unknown command {command}."
        return _DTypeCommandDispatcher.SUPPORTED[command]


class LayerNormParser(OpCLIParser):
    def get_signature(args: argparse.Namespace) -> LayerNormSignature:
        shape = _parse_shape(args.input)
        # Apparently the MIOpen driver can only normalize one dimension, and seems
        # to be imprecise if it is not the last dimension.
        assert (
            args.normalized_dim == len(shape) - 1
        ), "Can only normalize one trailing dimension for now (MIOpen limitation)."
        normalized_shape = shape[args.normalized_dim :]

        match args.forw:
            case 1:
                mode = Mode.FORWARD
            case 2:
                mode = Mode.INPUT_BACKWARD
            case 3:
                mode = Mode.WEIGHT_BACKWARD
            case 4:
                mode = Mode.BIAS_BACKWARD
            case _:
                raise ValueError(f"Unsupported mode {args.forw}.")

        return LayerNormSignature(
            input_shape=shape,
            normalized_shape=normalized_shape,
            eps=args.eps,
            elementwise_affine=(args.mode == 0),
            bias=True,
            dtype=_DTypeCommandDispatcher.get_dtype(args.command),
            mode=mode,
        )

    def get_miopen_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command", default="layernorm", choices=_DTypeCommandDispatcher.choices()
        )
        parser.add_argument(
            "--forw", "-F", type=int, default=1, help="Run only forward LayerNorm"
        )
        parser.add_argument(
            "--input",
            "-X",
            type=str,
            help="Input Tensor descriptor.\nFormat: NxC[xD]xHxW",
        )
        parser.add_argument("--eps", "-e", type=float, default=1e-5, help="Alpha")
        parser.add_argument(
            "--mode",
            "-m",
            type=int,
            default=0,
            choices=[0, 1],
            help="elemwise affine mode (0), weight and bias mode (1)",
        )
        parser.add_argument(
            "--normalized_dim", "-o", type=int, default=3, help="Normalized dim"
        )
        return parser

    @classmethod
    def get_op_name(cls) -> str:
        return "layernorm"
