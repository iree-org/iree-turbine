# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import tempfile
import pytest

from typing import Sequence
from pathlib import Path

import torch

from iree.turbine.kernel.boo.fusion import fusion_transform, OpFusionSpec, FusionSchema
from iree.turbine.kernel.boo.runtime import set_cache_dir


class SampleModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class SampleModule1(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        r0 = self.relu(x0)
        r1 = self.relu(x1)
        a0 = r0 + r1
        # Sum in both orderings to verify node_list collection is valid.
        a1 = a0 + r1
        a2 = r1 + a0
        a3 = a1 + a2
        mm = self.linear(a3)
        return mm


class SampleModule2(torch.nn.Module):
    def __init__(self, num_features: int = 3, kernel_size: int | Sequence[int] = 1):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
            ),
            torch.nn.Sigmoid(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return self.layer1(self.layer0(x))


class SampleModule3(torch.nn.Module):
    def __init__(self, num_features: int = 3, kernel_size: int | Sequence[int] = 1):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
            ),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=kernel_size,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return self.layer1(self.layer0(x))


class SubgraphReplacementTest(unittest.TestCase):
    def testReplacementWithPytorchBackward(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule(in_features=16, out_features=32)
            x = torch.ones([3, 4, 16, 16])

            fusion_schema: FusionSchema = {
                torch.ops.aten.linear.default: OpFusionSpec(
                    consumers=(torch.ops.aten.relu.default,)
                ),
            }
            fused_m = fusion_transform(m, (x,), fusion_schema=fusion_schema)

            fused_m_print = str(fused_m)

            self.assertNotIn("torch.ops.aten.linear", fused_m_print)

            self.assertEqual(
                fused_m.linear.weight.data_ptr(), m.linear.weight.data_ptr()
            )
            self.assertEqual(fused_m.linear.bias.data_ptr(), m.linear.bias.data_ptr())

            y = fused_m(x)

            y.sum().backward()

            self.assertIsNotNone(fused_m.linear.weight.grad)
            self.assertIsNotNone(fused_m.linear.bias.grad)

            x2 = torch.ones([3, 3, 32, 16])

            with self.assertRaises(RuntimeError):
                y2 = fused_m(x2)

    def testReplacementRecursiveFusion(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule1(in_features=16, out_features=16)
            x0 = torch.randn([16, 16])
            x1 = torch.randn([16, 16])
            schema: FusionSchema = {
                torch.ops.aten.linear.default: OpFusionSpec(
                    recursive=True,
                    producers=(torch.ops.aten.relu.default, torch.ops.aten.add.Tensor),
                )
            }
            fused_m = fusion_transform(m, (x0, x1), fusion_schema=schema)
            self.assertIn("generated_autograd_boo_fused_op_", str(fused_m))
            self.assertNotIn("torch.ops.aten.relu", str(fused_m))
            self.assertNotIn("torch.ops.aten.linear", str(fused_m))
            self.assertNotIn("torch.ops.aten.add", str(fused_m))
            y = fused_m(x0, x1)
            self.assertEqual(list(y.shape), [16, 16])

    def testReplacementChannelsLastConv(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule3().to(memory_format=torch.channels_last)
            x = torch.ones([2, 3, 16, 16], requires_grad=False)
            schema: FusionSchema = {
                torch.ops.aten.conv2d.default: OpFusionSpec(
                    recursive=True,
                    consumers=(torch.ops.aten.sigmoid.default,),
                )
            }
            fused_m = fusion_transform(m, (x,), fusion_schema=schema)
            self.assertNotIn("torch.ops.aten.", str(fused_m))
            y = fused_m(x)

    def testReplacementNonRecursiveFusion(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule1(in_features=16, out_features=16)
            x0 = torch.randn([16, 16])
            x1 = torch.randn([16, 16])
            schema: FusionSchema = {
                torch.ops.aten.linear.default: OpFusionSpec(
                    recursive=False,
                    producers=(torch.ops.aten.relu.default, torch.ops.aten.add.Tensor),
                )
            }
            fused_m = fusion_transform(m, (x0, x1), fusion_schema=schema)
            self.assertNotIn("torch.ops.aten.linear", str(fused_m))
            self.assertIn("generated_autograd_boo_fused_op_", str(fused_m))
            self.assertIn("torch.ops.aten.relu", str(fused_m))
            y = fused_m(x0, x1)
            self.assertEqual(list(y.shape), [16, 16])

    @pytest.mark.xfail(
        reason=(
            "Bug with repeated replacement when the first matching graph does not require the same combination of input gradients. "
            "Remove when issue https://github.com/iree-org/iree-turbine/issues/1014 is resolved."
        )
    )
    def testRepeatedReplacementBackward(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule2()
            x = torch.ones([2, 3, 16, 16], requires_grad=False)
            schema: FusionSchema = {
                torch.ops.aten.conv2d.default: OpFusionSpec(
                    recursive=True,
                    consumers=(torch.ops.aten.sigmoid.default,),
                )
            }
            fused_m = fusion_transform(m, (x,), fusion_schema=schema)
            self.assertNotIn("torch.ops.aten.", str(fused_m))
            y = fused_m(x)
            y.sum().backward()

            self.assertIsInstance(
                fused_m.get_parameter("layer0.0.weight").grad,
                torch.Tensor,
                f"Expected `layer0.0.weight.grad` to be a torch.Tensor, got {type(fused_m.get_parameter('layer0.0.weight').grad)}",
            )
            self.assertIsInstance(
                fused_m.get_parameter("layer0.0.bias").grad,
                torch.Tensor,
                f"Expected `layer0.0.bias.grad` to be a torch.Tensor, got {type(fused_m.get_parameter('layer0.0.bias').grad)}",
            )
            found_zero_grad = False
            err_msg = ""
            for name, param in m.named_parameters():
                param_grad_is_zero = param.grad.abs().sum().item() == 0.0
                if param_grad_is_zero:
                    found_zero_grad = True
                    err_msg += f"Parameter {name} unexpectedly has zero gradient.\n"

            self.assertFalse(found_zero_grad, msg=err_msg)


if __name__ == "__main__":
    unittest.main()
