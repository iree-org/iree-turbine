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
from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.profiler import profile, ProfilerActivity

from iree.turbine.dynamo.backends import boo
from iree.turbine.kernel.boo.fusion import OpFusionSpec, FusionSchema
from iree.turbine.kernel.boo.runtime import (
    set_cache_dir,
    LaunchableRuntimeCache,
)


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
                torch.ops.aten.addmm.default: OpFusionSpec(
                    recursive=True,
                    consumers=(
                        torch.ops.aten.relu.default,
                        torch.ops.aten.view.default,
                    ),
                ),
            }

            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m,
                backend=boo.backend(
                    fusion_schema=fusion_schema, nested_backend=recorder
                ),
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

    def testReplacementWithRecompile(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule(in_features=16, out_features=32)
            x1 = torch.ones([3, 4, 16, 16])
            x2 = torch.ones([3, 4, 32, 16])

            fusion_schema: FusionSchema = {
                torch.ops.aten.addmm.default: OpFusionSpec(
                    recursive=True,
                    consumers=(
                        torch.ops.aten.relu.default,
                        torch.ops.aten.view.default,
                    ),
                ),
            }

            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m,
                dynamic=False,
                backend=boo.backend(
                    fusion_schema=fusion_schema, nested_backend=recorder
                ),
            )

            y1 = compiled_m(x1)

            compiled_m.eval()
            with torch.no_grad():
                y2 = compiled_m(x2)

            [gm1, gm2] = recorder.graphs

            outputs1 = gm1.graph.find_nodes(op="output")
            self.assertEqual(len(outputs1), 1)
            output_node1 = outputs1[0]
            # We aren't in inference mode for the first application.
            # This graph should return three outputs: (linear result, pre-transposed result, None)
            self.assertEqual(len(output_node1.args[0]), 3)

            outputs2 = gm2.graph.find_nodes(op="output")
            self.assertEqual(len(outputs2), 1)
            # The second application should not have the extra outputs being stashed for backwards.
            output_node2 = outputs2[0]
            self.assertEqual(len(output_node2.args[0]), 1)

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
                m, backend=boo.backend(fusion_schema=schema, nested_backend=recorder)
            )
            assert isinstance(compiled_m, torch.nn.Module)

            y = compiled_m(x0, x1)

            [fused_m] = recorder.graphs
            self.assertIn("torch.ops.boo.fused_op_", str(fused_m))
            self.assertNotIn("torch.ops.aten.relu", str(fused_m))
            self.assertNotIn("torch.ops.aten.addmm", str(fused_m))
            self.assertNotIn("torch.ops.aten.add.Tensor", str(fused_m))
            self.assertEqual(list(y.shape), [16, 16])

    def testReplacementChannelsLastConv(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            m = SampleModule3().to(memory_format=torch.channels_last)
            x = torch.ones([2, 3, 16, 16], requires_grad=False)
            schema: FusionSchema = {
                torch.ops.aten.convolution.default: OpFusionSpec(
                    recursive=True,
                    consumers=(torch.ops.aten.sigmoid.default,),
                )
            }
            expected_y = m(x)
            recorder = EagerAndRecordGraphs()
            compiled_m = torch.compile(
                m, backend=boo.backend(fusion_schema=schema, nested_backend=recorder)
            )
            y = compiled_m(x)
            [fwd_gm] = recorder.graphs
            self.assertNotIn("torch.ops.aten.", str(fwd_gm))
            self.assertEqual(list(y.shape), list(expected_y.shape))
            self.assertEqual(list(y.stride()), list(expected_y.stride()))

    def testReplacementSingleDispatch(self):
        LaunchableRuntimeCache.clear()
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "testReplacementSingleDispatch"
            set_cache_dir(cache_dir)
            x = torch.ones([2, 3, 16, 16], requires_grad=False)
            w = torch.ones([4, 3, 1, 1], requires_grad=False)
            schema: FusionSchema = {
                torch.ops.aten.convolution.default: OpFusionSpec(
                    recursive=True,
                    make_single_dispatch=True,
                    consumers=(torch.ops.aten.sigmoid.default,),
                )
            }

            @torch.compile(backend=boo.backend(fusion_schema=schema))
            def forward(x, w):
                return torch.ops.aten.sigmoid(torch.ops.aten.conv2d(x, w, None))

            y = forward(x, w)
            cached_items = list(cache_dir.glob("*/*.mlir"))
            self.assertEqual(
                len(cached_items),
                1,
                msg=f"Expected one cached items, got {cached_items}.",
            )
            mlir_file = cached_items[0]
            name = mlir_file.stem
            contents = mlir_file.read_text()
            self.assertIn(
                "make-single-dispatch",
                contents,
                msg=f"Expected single dispatch for kernel {name}.",
            )

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
                m, backend=boo.backend(fusion_schema=schema, nested_backend=recorder)
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
                m, backend=boo.backend(fusion_schema=schema, nested_backend=recorder)
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
    def testNestedCompilerInductor(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            device = torch.device("cuda:0")

            schema: FusionSchema = {
                torch.ops.aten.convolution.default: OpFusionSpec(
                    recursive=True, consumers=(torch.ops.aten.sigmoid.default,)
                )
            }

            mx = SampleModule3().to(device=device)
            mx.eval()
            my = SampleModule3().to(device=device)
            my.eval()

            @torch.compile(
                dynamic=False,
                backend=boo.backend(nested_backend="inductor", fusion_schema=schema),
            )
            def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x1 = mx(x)
                y1 = my(y)
                z = (x1 + y1 + 2.0) / 16.0
                return z

            gen = torch.Generator(device=device)
            gen.manual_seed(13)
            x = torch.randn([2, 3, 16, 16], generator=gen, device=device)
            y = torch.randn([1, 3, 16, 16], generator=gen, device=device)

            # Do a warmup compile/run.
            with torch.no_grad():
                _ = f(x, y)

            # Profile the next run.
            with torch.no_grad() and profile(
                activities=[ProfilerActivity.CUDA]
            ) as prof:
                z = f(x, y)

            key_averages = str(prof.key_averages())
            self.assertIn("fused_op", key_averages)
            self.assertIn("triton_poi_fused_add_div", key_averages)


if __name__ == "__main__":
    unittest.main()
