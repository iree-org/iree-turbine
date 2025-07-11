# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import torch
import torch.nn as nn

from iree.turbine import aot
import iree.turbine.ops._jinja_test_ops as ops


class CustomLinearTrailingOptional(nn.Module):
    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> torch.Tensor:
        return ops.test_linear_trailing_optional(input, weight, bias)


class CustomLinearMiddleOptional(nn.Module):
    # The somewhat odd spelling here---with *args instead of explicit
    # arguments---is to work around the fact that the fx importer doesn't
    # currently support constant arguments. For example, a ConstantArgument(name='bias',
    # value=None) in the fx graph will crash the fx -> mlir import currently.
    def forward(
        self,
        *args,
    ) -> torch.Tensor:
        input = args[0]
        weight = args[-1]
        bias = None if len(args) == 2 else args[1]
        return ops.test_linear_middle_optional(input, bias, weight)


class OptionalBiasTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(42)

        self.inputs = torch.randn((2, 4), generator=g)
        self.weight = torch.randn((4, 3), generator=g)
        self.bias = torch.randn((3), generator=g)

    def test_trailing_optional_with_bias(self):
        custom_module = CustomLinearTrailingOptional()
        custom_result = custom_module(self.inputs, self.weight, self.bias)
        self.assertTrue(
            torch.allclose(custom_result, self.inputs @ self.weight + self.bias)
        )

    def test_trailing_optional_without_bias(self):
        custom_module = CustomLinearTrailingOptional()
        custom_result = custom_module(self.inputs, self.weight)
        self.assertTrue(torch.allclose(custom_result, self.inputs @ self.weight))

    def test_middle_optional_with_bias(self):
        custom_module = CustomLinearMiddleOptional()
        custom_result = custom_module(self.inputs, self.bias, self.weight)
        self.assertTrue(
            torch.allclose(custom_result, self.inputs @ self.weight + self.bias)
        )

    def test_middle_optional_without_bias(self):
        custom_module = CustomLinearMiddleOptional()
        custom_result = custom_module(self.inputs, self.weight)
        self.assertTrue(torch.allclose(custom_result, self.inputs @ self.weight))

    def test_aot_middle_optional_with_bias(self):
        e = aot.export(
            CustomLinearMiddleOptional(), args=(self.inputs, self.bias, self.weight)
        )
        mlir_asm = str(e.mlir_module)
        self.assertIn(
            "util.call @turbine_test_linear_middle_optional_2d_f32_biased(%0, %1, %2)",
            mlir_asm,
        )
        self.assertIn("linalg.matmul", mlir_asm)
        self.assertIn("linalg.generic", mlir_asm)

    def test_aot_middle_optional_without_bias(self):
        e = aot.export(CustomLinearMiddleOptional(), args=(self.inputs, self.weight))
        mlir_asm = str(e.mlir_module)
        self.assertIn("util.call @turbine_test_linear_2d_f32(%0, %1)", mlir_asm)
        self.assertIn("linalg.matmul", mlir_asm)
        self.assertNotIn("linalg.generic", mlir_asm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
