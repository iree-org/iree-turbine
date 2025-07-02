# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import torch

from iree.turbine.kernel.boo.gemm_exports.gemm import GEMMSignature
from iree.turbine.kernel.boo.exports.parser import OpCLIParser


class _DTypeCommandDispatcher:
    SUPPORTED = {
        "gemm": torch.float,
        "gemmfp16": torch.float16,
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


class GEMMParser(OpCLIParser):
    @staticmethod
    def get_signature(args: argparse.Namespace) -> GEMMSignature:
        assert (
            args.a_w > 0
        ), f"Expected positive value for A width, got shape {args.a_w}"
        assert (
            args.a_h > 0
        ), f"Expected positive value for A height, got shape {args.a_h}"
        assert (
            args.b_w > 0
        ), f"Expected positive value for B width, got shape {args.b_w}"

        return GEMMSignature(
            a_shape=[args.a_w, args.a_h],
            b_shape=[args.a_h, args.b_w],
            transpose_a=args.transA,
            transpose_b=args.transB,
            dtype=_DTypeCommandDispatcher.get_dtype(args.command),
        )

    @staticmethod
    def get_miopen_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "command", default="gemm", choices=_DTypeCommandDispatcher.choices()
        )
        parser.add_argument(
            "--a_w", "-k", type=int, default=256, help="Width of A matrix (Default=256)"
        )
        parser.add_argument(
            "--a_h",
            "-m",
            type=int,
            default=256,
            help="Height of A matrix (Default=256)",
        )
        parser.add_argument(
            "--b_w", "-n", type=int, default=256, help="Width of B matrix (Default=256)"
        )
        parser.add_argument(
            "--transA", "-u", type=int, default=0, help="Transpose A matrix (Default=0)"
        )
        parser.add_argument(
            "--transB", "-v", type=int, default=0, help="Transpose B matrix (Default=0)"
        )
        return parser
