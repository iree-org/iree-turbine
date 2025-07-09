# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import tempfile

from typing import Sequence
from pathlib import Path

import torch
from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs

from iree.turbine.dynamo.backends import boo
from iree.turbine.kernel.boo.fusion import OpFusionSpec, FusionSchema
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


class SubgraphReplacementTest(unittest.TestCase):
    def testReplacementWithPytorchBackward(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule(in_features=16, out_features=32)
            x = torch.ones([3, 4, 16, 16])

            fusion_schema: FusionSchema = {
                torch.ops.aten.addmm.default: OpFusionSpec(
                    consumers=(torch.ops.aten.relu.default, torch.ops.aten.view.default)
                ),
            }

            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m, backend=boo.backend(fusion_schema=fusion_schema, backend=recorder)
            )
            assert isinstance(compiled_m, torch.nn.Module)

            y = compiled_m(x)

            [fused_m] = recorder.graphs
            assert isinstance(fused_m, fx.GraphModule)
            fused_m_print = str(fused_m)

            self.assertIn("torch.ops.boo.fused_op_", fused_m_print)
            self.assertNotIn("torch.ops.aten.addmm", fused_m_print)
            self.assertNotIn("torch.ops.aten.relu", fused_m_print)

            self.assertEqual(
                compiled_m.linear.weight.data_ptr(), m.linear.weight.data_ptr()
            )
            self.assertEqual(
                compiled_m.linear.bias.data_ptr(), m.linear.bias.data_ptr()
            )

            y.sum().backward()

            self.assertIsNotNone(compiled_m.linear.weight.grad)
            self.assertIsNotNone(compiled_m.linear.bias.grad)

            x2 = torch.ones([3, 3, 32, 16])

            with self.assertRaises(RuntimeError):
                y2 = compiled_m(x2)

    def testReplacementRecursiveFusion(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule1(in_features=16, out_features=16)
            x0 = torch.randn([16, 16])
            x1 = torch.randn([16, 16])
            schema: FusionSchema = {
                torch.ops.aten.addmm.default: OpFusionSpec(
                    recursive=True,
                    producers=(torch.ops.aten.relu.default, torch.ops.aten.add.Tensor),
                )
            }
            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m, backend=boo.backend(fusion_schema=schema, backend=recorder)
            )
            assert isinstance(compiled_m, torch.nn.Module)

            y = compiled_m(x0, x1)

            [fused_m] = recorder.graphs
            self.assertIn("torch.ops.boo.fused_op_", str(fused_m))
            self.assertNotIn("torch.ops.aten.relu", str(fused_m))
            self.assertNotIn("torch.ops.aten.addmm", str(fused_m))
            self.assertNotIn("torch.ops.aten.add.Tensor", str(fused_m))
            self.assertEqual(list(y.shape), [16, 16])

    def testReplacementNonRecursiveFusion(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule1(in_features=16, out_features=16)
            x0 = torch.randn([16, 16])
            x1 = torch.randn([16, 16])
            schema: FusionSchema = {
                torch.ops.aten.addmm.default: OpFusionSpec(
                    recursive=False,
                    producers=(torch.ops.aten.relu.default, torch.ops.aten.add.Tensor),
                )
            }
            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m, backend=boo.backend(fusion_schema=schema, backend=recorder)
            )
            assert isinstance(compiled_m, torch.nn.Module)
            y = compiled_m(x0, x1)
            [fused_m] = recorder.graphs

            self.assertNotIn("torch.ops.aten.addmm", str(fused_m))
            self.assertIn("torch.ops.boo.fused_op_", str(fused_m))
            self.assertIn("torch.ops.aten.relu", str(fused_m))
            self.assertEqual(list(y.shape), [16, 16])

    def testRepeatedReplacementBackward(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule2()
            x = torch.ones([2, 3, 16, 16], requires_grad=False)
            schema: FusionSchema = {
                torch.ops.aten.convolution.default: OpFusionSpec(
                    recursive=True,
                    consumers=(torch.ops.aten.sigmoid.default,),
                )
            }

            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m, backend=boo.backend(fusion_schema=schema, backend=recorder)
            )

            y = compiled_m(x)
            y.sum().backward()

            # Only the backward module should contain aten ops.
            forward_module, backward_module = recorder.graphs
            self.assertIn("torch.ops.aten.", str(backward_module))
            self.assertNotIn("torch.ops.aten.", str(forward_module))

            self.assertIsInstance(
                compiled_m.get_parameter("layer0.0.weight").grad,
                torch.Tensor,
                f"Expected `layer0.0.weight.grad` to be a torch.Tensor, got {type(compiled_m.get_parameter('layer0.0.weight').grad)}",
            )
            self.assertIsInstance(
                compiled_m.get_parameter("layer0.0.bias").grad,
                torch.Tensor,
                f"Expected `layer0.0.bias.grad` to be a torch.Tensor, got {type(compiled_m.get_parameter('layer0.0.bias').grad)}",
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
