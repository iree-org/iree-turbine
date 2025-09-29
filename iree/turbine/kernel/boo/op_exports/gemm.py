# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from enum import IntEnum
from typing import Any
import math
import argparse

from ..exports.signature import OpSignature, ModeBase
from ..exports.parser import OpCLIParser


class Mode(ModeBase, IntEnum):
    """Mode selector for GEMM, with each gradient being its own mode."""

    FORWARD = 0
    A_BACKWARD = 1
    B_BACKWARD = 2


class GEMMSignature(OpSignature):
    """Signature for General Matrix Multiplication (GEMM) operations"""

    a_shape: list[int]
    b_shape: list[int]
    transpose_a: bool = False
    transpose_b: bool = False
    dtype: torch.dtype = torch.float32
    mode: Mode = Mode.FORWARD

    def __init__(
        self,
        *,
        a_shape: list[int],
        b_shape: list[int],
        transpose_a: bool = False,
        transpose_b: bool = False,
        dtype: torch.dtype = torch.float32,
        mode: Mode = Mode.FORWARD,
    ):
        if len(a_shape) != len(b_shape) != 2:
            raise ValueError(f"expected shapes to be of rank 2")
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.dtype = dtype
        self.mode = mode

    def get_output_size(self) -> int:
        """Returns the size of the output tensor in bytes."""
        m = self.a_shape[0]
        n = self.b_shape[1]
        return math.prod([m, n]) * self.dtype.itemsize

    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        if self.mode == Mode.FORWARD:
            return GEMMForward(self)
        elif self.mode == Mode.A_BACKWARD:
            return GEMMBackwardA(self)
        elif self.mode == Mode.B_BACKWARD:
            return GEMMBackwardB(self)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen = gen.manual_seed(seed)

        def get(shape: list[int]) -> torch.Tensor:
            if splat_value is not None:
                return torch.ones(shape, dtype=self.dtype, device=device) * splat_value
            return torch.randn(shape, generator=gen, dtype=self.dtype, device=device)

        a_shape = self.a_shape
        if self.transpose_a:
            a_shape = [self.a_shape[1], self.a_shape[0]]
        b_shape = self.b_shape
        if self.transpose_b:
            b_shape = [self.b_shape[1], self.b_shape[0]]

        return (get(a_shape), get(b_shape))

    @property
    def is_forward(self) -> bool:
        return self.mode == Mode.FORWARD

    def arrange_backward_launch_args(self, forward_args, forward_results):
        # forward: C = A @ B
        if self.mode == Mode.A_BACKWARD:
            # backward: dA = dC @ B^T
            return (forward_args[1],)
        elif self.mode == Mode.B_BACKWARD:
            # backward: dB = A^T @ dC
            return (forward_args[0],)

    @property
    def func_name(self) -> str:
        """Format: gemm_{dtype}_{mode}_{m}x{k}x{n}[_transA][_transB]"""
        m, k = self.a_shape
        _, n = self.b_shape

        name_items = [
            "gemm",
            str(self.dtype).removeprefix("torch."),
            self.mode.name.lower(),
            f"{m}x{k}x{n}",
        ]
        if self.transpose_a:
            name_items.append("transA")
        if self.transpose_b:
            name_items.append("transB")
        return "_".join(name_items)

    def as_init_kwargs(self) -> dict[str, Any]:
        return {
            "a_shape": self.a_shape,
            "b_shape": self.b_shape,
            "transpose_a": self.transpose_a,
            "transpose_b": self.transpose_b,
            "dtype": self.dtype,
            "mode": self.mode,
        }


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

        match args.forw:
            case 1:
                mode = Mode.FORWARD
            case 2:
                mode = Mode.A_BACKWARD
            case 3:
                mode = Mode.B_BACKWARD
            case _:
                raise ValueError(f"Unsupported mode {args.forw}.")

        return GEMMSignature(
            a_shape=[args.a_w, args.a_h],
            b_shape=[args.a_h, args.b_w],
            transpose_a=args.transA,
            transpose_b=args.transB,
            dtype=_DTypeCommandDispatcher.get_dtype(args.command),
            mode=mode,
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
        parser.add_argument(
            "--forw",
            "-F",
            type=int,
            default=1,
            help="Run only Forward Gemm (Default=1)",
        )
        return parser


class GEMMForward(torch.nn.Module):
    """Forward GEMM operation"""

    def __init__(self, sig: GEMMSignature):
        super().__init__()
        self.sig = sig

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # C = A @ B
        if self.sig.transpose_a:
            a = a.t()
        if self.sig.transpose_b:
            b = b.t()
        return torch.mm(a, b)


class GEMMBackwardA(torch.nn.Module):
    """Backward for input A"""

    def __init__(self, sig: GEMMSignature):
        super().__init__()
        self.sig = sig

    def forward(self, grad_output: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # For C = A @ B, dA = dC @ B^T
        if self.sig.transpose_b:
            result = torch.mm(grad_output, b)
        else:
            result = torch.mm(grad_output, b.t())

        if self.sig.transpose_a:
            return result.t()

        return result


class GEMMBackwardB(torch.nn.Module):
    """Backward for input B"""

    def __init__(self, sig: GEMMSignature):
        super().__init__()
        self.sig = sig

    def forward(self, grad_output: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # For C = A @ B, dB = A^T @ dC
        if self.sig.transpose_a:
            result = torch.mm(a, grad_output)
        else:
            result = torch.mm(a.t(), grad_output)

        if self.sig.transpose_b:
            return result.t()

        return result
