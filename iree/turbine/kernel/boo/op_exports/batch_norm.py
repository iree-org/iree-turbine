# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Sequence, Optional, final
import torch
from ..exports.signature import OpSignature
from ..exports.parser import OpCLIParser


@final
@dataclass
class BatchNormSignature(OpSignature):
    """Batch normalization signature that provides information for launching specific kernels."""

    input_shape: list[int]
    input_dtype: torch.dtype

    @property
    def is_forward(self):
        return True

    @property
    def func_name(self):
        return "bnorm:" + ":".join(str(v) for v in asdict(self).values())

    def make_signature_copy_for_forward(self) -> "BatchNormSignature":
        raise NotImplementedError()

    def get_arg_index_for_backward(self) -> int | None:
        raise NotImplementedError()

    def arrange_backward_launch_args(
        self,
        forward_args: tuple[torch.Tensor, ...],
        forward_results: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor]:
        raise NotImplementedError()

    def as_init_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError()

    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        class AtenBatchNorm(torch.nn.Module):
            def forward(self, *args: torch.Tensor):
                training = True
                exponential_average_factor = 0.1
                epsilon = 1e-5
                return torch.ops.aten._native_batch_norm_legit_functional(
                    *args, training, exponential_average_factor, epsilon
                )

        return AtenBatchNorm()

    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device)
        if seed:
            gen = gen.manual_seed(seed)

        def get(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
            if splat_value is not None:
                return torch.ones(shape, dtype=dtype, device=device) * splat_value
            return torch.randn(shape, generator=gen, dtype=dtype, device=device)

        _, C, *_ = self.input_shape
        memory_format = torch.channels_last
        input = get(self.input_shape, dtype=self.input_dtype).to(
            memory_format=memory_format
        )
        weight = get((C,), dtype=torch.float32)
        bias = get((C,), dtype=torch.float32)
        running_mean = get((C,), dtype=torch.float32)
        running_var = get((C,), dtype=torch.float32)
        return (
            input,
            weight,
            bias,
            running_mean,
            running_var,
        )


class _DTypeCommandDispatcher:
    SUPPORTED = {
        "bnormbfp16": torch.bfloat16,
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


class BatchNormParser(OpCLIParser):
    @staticmethod
    def get_signature(args: argparse.Namespace) -> BatchNormSignature:
        if args.alpha != 1.0:
            raise NotImplementedError("Only alpha=1.0 is supported")
        if args.beta != 0.0:
            raise NotImplementedError("Only beta=0.0 is supported")
        if args.forw != 1:
            raise NotImplementedError("Only forw=1 (train) is supported")
        if args.back != 0:
            raise NotImplementedError("Only back=0 is supported")
        if args.mode != 0:
            raise NotImplementedError("Only mode=0 is supported")
        if args.run != 1:
            raise NotImplementedError("Only run=1 is supported")
        if args.save != 1:
            raise NotImplementedError("Only save=1 is supported")
        if args.layout != "NHWC":
            raise NotImplementedError("Only layout=NHWC is supported")
        input_shape = [args.batchsize, args.in_channels, args.in_h, args.in_w]
        if args.in_d:
            input_shape.append(args.in_d)
        return BatchNormSignature(
            input_shape=input_shape,
            input_dtype=_DTypeCommandDispatcher.get_dtype(args.command),
        )

    @staticmethod
    def get_miopen_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--alpha",
            "-A",
            type=float,
            default=1.0,
            help="Alpha (Default=1.0)",
        )
        parser.add_argument(
            "--beta",
            "-B",
            type=float,
            default=0.0,
            help="Beta (Default=0.)",
        )
        parser.add_argument(
            "--in_d", "-D", type=int, default=0, help="Input Depth (Default=0)"
        )
        parser.add_argument(
            "--forw",
            "-F",
            type=int,
            default=1,
            help="Run Forward Train (off: 0, train: 1, inference: 2) Batch Normalization (Default=1)",
        )
        parser.add_argument(
            "--in_h", "-H", type=int, default=32, help="Input Height (Default=32)"
        )
        parser.add_argument(
            "--in_w", "-W", type=int, default=32, help="Input Width (Default=32)"
        )
        parser.add_argument(
            "--back",
            "-b",
            type=int,
            default=0,
            help="Backwards Propagation (off: 0, on: 1) Batch Normalization (Default=0)",
        )
        parser.add_argument(
            "--in_channels",
            "-c",
            type=int,
            default=3,
            help="Number of Input Channels (Default=3)",
        )
        parser.add_argument(
            "--mode",
            "-m",
            type=int,
            default=0,
            help="Normalization Mode (per-activation (0) or spatial (1)) (Default=0)",
        )
        parser.add_argument(
            "--batchsize",
            "-n",
            type=int,
            default=32,
            help="Mini-batch size (Default=32)",
        )
        parser.add_argument(
            "--run",
            "-r",
            type=int,
            default=0,
            help="Keep running mean and variance, or on inference, use these values. (Default=0)",
        )
        parser.add_argument(
            "--save",
            "-s",
            type=int,
            default=0,
            help="Save off mean and inverse variance, or on backprop, use these values. (Default=0)",
        )
        parser.add_argument("--layout", "-L", type=str, default="NCHW", help="Layout")
        parser.add_argument("command", choices=_DTypeCommandDispatcher.choices())
        return parser

    @classmethod
    def get_op_name(cls) -> str:
        return "bnorm"
