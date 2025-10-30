# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ast
import base64
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import torch

from typing import Any
from typing_extensions import override
import argparse

from ..exports.signature import OpSignature
from ..exports.parser import OpCLIParser


@dataclass
class AtenSignature(OpSignature):
    """
    Generic signature for ATen ops. The values expected here match the values in
    'torch.profiler's chrome trace events, which look like:
      {
        "name": "aten::conv2d", ...
        "args": {
          "Input Dims": [[128, 3, 256, 256], [64, 3, 7, 7], [], [], [], [], []],
          "Input type": ["c10::BFloat16", "c10::BFloat16", "", "ScalarList", "ScalarList", "ScalarList", "Scalar"],
          "Input Strides": [[196608, 1, 768, 3], [147, 1, 21, 3], [], [], [], [], []],
          "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", "1"],
          ...
        }
      }
    """

    name: str
    input_dims: Sequence[Sequence[int]]
    input_type: Sequence[str]
    input_strides: Sequence[Sequence[int]]
    concrete_inputs: Sequence[str]

    @override
    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        # Translate e.g. "aten::convolution" -> torch.ops.aten.convolution
        [op_namespace, op_unqualified_name] = self.name.split("::")
        func = getattr(getattr(torch.ops, op_namespace), op_unqualified_name)

        class FuncModule(torch.nn.Module):
            def forward(self, *args):
                return func(*args)

        return FuncModule()

    @override
    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen = gen.manual_seed(seed)

        def get(
            dims: Sequence[int], type: str, strides: Sequence[int], concrete: str
        ) -> torch.Tensor:
            if concrete != "":
                raise NotImplementedError(
                    f"Concrete inputs not supported yet: {concrete}"
                )

            # Handle tensor arguments.
            match type:
                case "float":
                    dtype = float
                case "c10::BFloat16":
                    dtype = torch.bfloat16
                case _:
                    raise ValueError(f"Unsupported input type: {type}")

            val = (
                torch.full(dims, splat_value, dtype=dtype, device=device)
                if splat_value is not None
                else torch.randn(dims, generator=gen, dtype=dtype, device=device)
            )
            return torch.as_strided(val, dims, strides)

        return tuple(
            get(*args)
            for args in zip(
                self.input_dims,
                self.input_type,
                self.input_strides,
                self.concrete_inputs,
                strict=True,
            )
        )

    @property
    @override
    def is_forward(self) -> bool:
        raise NotImplementedError()

    @override
    def arrange_backward_launch_args(self, forward_args, forward_results):
        raise NotImplementedError()

    @property
    @override
    def func_name(self) -> str:
        # This name is used as a file system path, but the fields here may
        # contain special characters. A URL-safe b64 encode ensures only valid
        # characters are used.
        return base64.urlsafe_b64encode(
            (
                self.name
                + str(self.input_dims)
                + str(self.input_type)
                + str(self.input_strides)
                + str(self.concrete_inputs)
            ).encode()
        ).decode()

    @override
    def as_init_kwargs(self) -> dict[str, Any]:
        return asdict(self)


class AtenParser(OpCLIParser):
    @override
    @staticmethod
    def get_signature(args: argparse.Namespace) -> AtenSignature:
        return AtenSignature(
            name=args.name,
            input_dims=ast.literal_eval(args.input_dims),
            input_type=ast.literal_eval(args.input_type),
            input_strides=ast.literal_eval(args.input_strides),
            concrete_inputs=ast.literal_eval(args.concrete_inputs),
        )

    @override
    @staticmethod
    def get_miopen_parser() -> argparse.ArgumentParser:
        """Returns a pre-configured argument parser with MIOpen-compatible options."""
        parser = argparse.ArgumentParser()
        parser.add_argument("name")
        parser.add_argument("input_dims")
        parser.add_argument("input_type")
        parser.add_argument("input_strides")
        parser.add_argument("concrete_inputs")
        return parser
