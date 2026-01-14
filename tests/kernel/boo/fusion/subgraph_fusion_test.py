# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from typing import Sequence
from pathlib import Path

import torch
from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.profiler import profile, ProfilerActivity

from iree.turbine.dynamo.backends import boo
from iree.turbine.kernel.boo.fusion import OpFusionSpec, FusionSchema


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


class TestSubgraphReplacement:
    def testLinearLayoutHandling(self, boo_cache_dir: Path):
        schema: FusionSchema = {torch.ops.aten.addmm.default: OpFusionSpec()}
        recorder = EagerAndRecordGraphs()
        m = torch.compile(
            torch.nn.Linear(in_features=64, out_features=16),
            backend=boo.backend(
                nested_backend=recorder,
                fusion_schema=schema,
            ),
        )
        x = torch.randn([32, 64])
        m(x)
        mlir_files = list(boo_cache_dir.glob("*/*.mlir"))
        assert len(mlir_files) == 1, f"Expected one mlir file, got {mlir_files}."
        mlir_string = mlir_files[0].read_text()
        assert "torch.aten.permute" in mlir_string

    def testReplacementWithPytorchBackward(self):
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
            backend=boo.backend(fusion_schema=fusion_schema, nested_backend=recorder),
        )
        assert isinstance(compiled_m, torch.nn.Module)

        y = compiled_m(x)

        [fused_m] = recorder.graphs
        assert isinstance(fused_m, fx.GraphModule)
        fused_m_print = str(fused_m)

        assert "torch.ops.boo.fused_op_" in fused_m_print
        assert "torch.ops.aten.addmm" not in fused_m_print
        assert "torch.ops.aten.relu" not in fused_m_print

        assert compiled_m.linear.weight.data_ptr() == m.linear.weight.data_ptr()
        assert compiled_m.linear.bias.data_ptr() == m.linear.bias.data_ptr()

        y.sum().backward()

        assert compiled_m.linear.weight.grad is not None
        assert compiled_m.linear.bias.grad is not None

    def testReplacementWithRecompile(self):
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
            backend=boo.backend(fusion_schema=fusion_schema, nested_backend=recorder),
        )

        y1 = compiled_m(x1)

        compiled_m.eval()
        with torch.no_grad():
            y2 = compiled_m(x2)

        [gm1, gm2] = recorder.graphs

        outputs1 = gm1.graph.find_nodes(op="output")
        assert len(outputs1) == 1
        output_node1 = outputs1[0]
        # We aren't in inference mode for the first application.
        # This graph should return three outputs: (linear result, pre-transposed result, None)
        assert len(output_node1.args[0]) == 3

        outputs2 = gm2.graph.find_nodes(op="output")
        assert len(outputs2) == 1
        # The second application should not have the extra outputs being stashed for backwards.
        output_node2 = outputs2[0]
        assert len(output_node2.args[0]) == 1

    def testReplacementRecursiveFusion(self):
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
        assert "torch.ops.boo.fused_op_" in str(fused_m)
        assert "torch.ops.aten.relu" not in str(fused_m)
        assert "torch.ops.aten.addmm" not in str(fused_m)
        assert "torch.ops.aten.add.Tensor" not in str(fused_m)
        assert list(y.shape) == [16, 16]

    def testReplacementChannelsLastConv(self):
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
        for node in fwd_gm.graph.nodes:
            if not node.op == "call_function":
                continue
            if not str(node.target).startswith("aten."):
                continue
            assert (
                str(node.target) == "aten.detach.default"
            ), f"Got an unexpected op {node} in graph:\n {fwd_gm.print_readable(print_output=False)}."

        assert list(y.shape) == list(expected_y.shape)
        assert list(y.stride()) == list(expected_y.stride())

    def testReplacementMultiOutputNode(self):
        schema: FusionSchema = {
            torch.ops.aten._native_batch_norm_legit_functional.default: OpFusionSpec(),
        }
        recorder = EagerAndRecordGraphs()
        backend = boo.backend(nested_backend=recorder, fusion_schema=schema)
        m = torch.compile(torch.nn.BatchNorm2d(num_features=16), backend=backend)

        x = torch.randn([2, 16, 32, 32])

        y = m(x)

        assert len(recorder.graphs) == 1, "Expected one graph."
        assert "torch.ops.boo.fused_op" in str(
            recorder.graphs[0]
        ), "Expected a boo op in graph."
        assert m.num_batches_tracked == 1, "Expected one tracked batch."

        y.sum().backward()

        assert len(recorder.graphs) == 2, "Expected two graphs after backward call."
        assert "torch.ops.boo" not in str(
            recorder.graphs[-1]
        ), "Expected no boo ops in backward graph."
        assert (
            isinstance(m.weight.grad, torch.Tensor)
            and torch.max(torch.abs(m.weight.grad)).item() != 0
        ), "Expected some weight gradient."
        assert (
            isinstance(m.bias.grad, torch.Tensor)
            and torch.max(torch.abs(m.bias.grad)).item() != 0
        ), "Expected some bias gradient."

    def testReplacementSingleDispatch(self, boo_cache_dir: Path):
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
        cached_items = list(boo_cache_dir.glob("*/*.mlir"))
        assert len(cached_items) == 1, f"Expected one cached items, got {cached_items}."
        mlir_file = cached_items[0]
        name = mlir_file.stem
        contents = mlir_file.read_text()
        assert (
            "make-single-dispatch" in contents
        ), f"Expected single dispatch for kernel {name}."

    def testReplacementNonRecursiveFusion(self):
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

        assert "torch.ops.aten.addmm" not in str(fused_m)
        assert "torch.ops.boo.fused_op_" in str(fused_m)
        assert "torch.ops.aten.relu" in str(fused_m)
        assert list(y.shape) == [16, 16]

    def testRepeatedReplacementBackward(self):
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
        for node in forward_module.graph.nodes:
            if node.op != "call_function":
                continue
            if str(node.target).startswith("aten."):
                assert (
                    str(node.target) == "aten.detach.default"
                ), f"Got unexpected node: {node}.\nSee:\n{forward_module.print_readable(print_output=False)}."

        assert not any(
            [
                str(node.target).startswith("boo.")
                for node in backward_module.graph.nodes
                if node.target == "call_function"
            ]
        ), "Expected no boo ops for backward graph."

        assert isinstance(
            compiled_m.get_parameter("layer0.0.weight").grad, torch.Tensor
        ), f"Expected `layer0.0.weight.grad` to be a torch.Tensor, got {type(compiled_m.get_parameter('layer0.0.weight').grad)}"
        assert isinstance(
            compiled_m.get_parameter("layer0.0.bias").grad, torch.Tensor
        ), f"Expected `layer0.0.bias.grad` to be a torch.Tensor, got {type(compiled_m.get_parameter('layer0.0.bias').grad)}"
        found_zero_grad = False
        err_msg = ""
        for name, param in m.named_parameters():
            param_grad_is_zero = param.grad.abs().sum().item() == 0.0
            if param_grad_is_zero:
                found_zero_grad = True
                err_msg += f"Parameter {name} unexpectedly has zero gradient.\n"

        assert not found_zero_grad, err_msg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
    def testNestedCompilerInductor(self):
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
        with torch.no_grad() and profile(activities=[ProfilerActivity.CUDA]) as prof:
            z = f(x, y)

        key_averages = str(prof.key_averages())
        assert "fused_op" in key_averages
        assert "triton_poi_fused_add_div" in key_averages
