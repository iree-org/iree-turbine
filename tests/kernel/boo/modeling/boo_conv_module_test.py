# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import tempfile
from pathlib import Path
import pytest

import torch

from iree.turbine.kernel.boo.ops import enable_backward, disable_backward
from iree.turbine.kernel.boo.modeling import (
    BooConv1d,
    BooConv2d,
    BooConv3d,
    replace_conv2d_with_boo_conv,
    replace_convs_with_boo,
)
from iree.turbine.kernel.boo.runtime import (
    set_cache_dir,
    LaunchableRuntimeCache,
)


@pytest.fixture(scope="class")
def use_backward():
    enable_backward()
    yield
    disable_backward()


class BooConvReplacementTest(unittest.TestCase):
    def setUp(self):
        LaunchableRuntimeCache.clear()
        LaunchableRuntimeCache.set_cache_limit(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv3d(
                    in_channels=3, out_channels=2, kernel_size=3, bias=False
                )
                self.conv1 = torch.nn.Conv2d(
                    in_channels=2, out_channels=3, kernel_size=2, bias=True
                )
                self.conv2 = torch.nn.Conv1d(
                    in_channels=3, out_channels=2, kernel_size=1, bias=False
                )

            def forward(self, x):
                return self.conv2(
                    self.conv1(self.conv0(x).flatten(-1, -2)).flatten(-1, -2)
                )

        self.model1 = M().to(device=self.device)

    def testAllReplacements(self):
        model = replace_convs_with_boo(self.model1)
        for name, module in model.named_modules():
            msg = f"Module {name} encountered improper conversion."
            self.assertNotIsInstance(module, torch.nn.Conv1d, msg=msg)
            self.assertNotIsInstance(module, torch.nn.Conv2d, msg=msg)
            self.assertNotIsInstance(module, torch.nn.Conv3d, msg=msg)

    def testOnly2dReplacement(self):
        model = replace_conv2d_with_boo_conv(self.model1)
        for name, module in model.named_modules():
            msg = f"Module {name} encountered improper conversion."
            self.assertNotIsInstance(module, BooConv1d, msg=msg)
            self.assertNotIsInstance(module, torch.nn.Conv2d, msg=msg)
            self.assertNotIsInstance(module, BooConv3d, msg=msg)

    def test1dAnd2dReplacement(self):
        model = replace_convs_with_boo(self.model1, allowed_spatial_dims=(1, 3))
        for name, module in model.named_modules():
            msg = f"Module {name} encountered improper conversion."
            self.assertNotIsInstance(module, torch.nn.Conv1d, msg=msg)
            self.assertNotIsInstance(module, BooConv2d, msg=msg)
            self.assertNotIsInstance(module, torch.nn.Conv3d, msg=msg)


class BooConvModuleTest(unittest.TestCase):
    def testBooConv1dParameters(self):
        m = BooConv1d(9, 2, 6, 1, 2, 2, 3, True)
        self.assertEqual(tuple(m.weight.shape), (2, 3, 6))
        self.assertIsNotNone(m.bias)
        self.assertEqual(tuple(m.bias.shape), (2,))

    def testBooConv2dParameters(self):
        m = BooConv2d(9, 2, 6, 1, 2, 2, 3, True)
        self.assertEqual(tuple(m.weight.shape), (2, 3, 6, 6))
        self.assertIsNotNone(m.bias)
        self.assertEqual(tuple(m.bias.shape), (2,))
        m2 = BooConv2d(10, 2, [2, 6], 1, 1, 1, 1, False)
        self.assertEqual(tuple(m2.weight.shape), (2, 10, 2, 6))
        self.assertIsNone(m2.bias)

    def testBooConv3dParameters(self):
        m = BooConv3d(10, 2, 6, 1, 2, 2, 1, True)
        self.assertEqual(tuple(m.weight.shape), (2, 10, 6, 6, 6))


@pytest.mark.usefixtures("use_backward")
class BooConv2dLaunchingTest(unittest.TestCase):
    def setUp(self):
        LaunchableRuntimeCache.clear()
        LaunchableRuntimeCache.set_cache_limit(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model0 = BooConv2d(
            in_channels=2, out_channels=3, kernel_size=2, bias=False
        ).to(device=self.device)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = torch.nn.Conv2d(
                    in_channels=3, out_channels=2, kernel_size=3, bias=False
                )
                self.conv1 = torch.nn.Conv2d(
                    in_channels=2, out_channels=3, kernel_size=2, bias=True
                )

            def forward(self, x):
                return self.conv1(self.conv0(x))

        self.model1 = M().to(device=self.device)

    def testBasic(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones([10, 2, 16, 16], device=self.device, dtype=torch.float32)
            _ = self.model0(x)
            self.assertIn(
                "conv_2d_float32_forward_10x2x16x16_nchw_3x2x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                [i.name for i in cache_dir.glob("*")],
            )

    def testNoBatch(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones([2, 16, 16], device=self.device, dtype=torch.float32)
            _ = self.model0(x)
            self.assertIn(
                "conv_2d_float32_forward_1x2x16x16_nchw_3x2x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                [i.name for i in cache_dir.glob("*")],
            )

    def testReplacement(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones([10, 3, 16, 16], device=self.device, dtype=torch.float32)
            model2 = replace_conv2d_with_boo_conv(self.model1)
            _ = model2(x)
            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x3x16x16_nchw_2x3x3x3_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_forward_b_10x2x14x14_nchw_3x2x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                func_names,
            )

    def testChannelsLast(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones([10, 2, 16, 16], device=self.device, dtype=torch.float32).to(
                memory_format=torch.channels_last
            )
            model = self.model0.to(memory_format=torch.channels_last)
            _ = model(x)
            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )

    def testReplacementChannelsLast(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones([10, 3, 16, 16], device=self.device, dtype=torch.float32).to(
                memory_format=torch.channels_last
            )
            model2 = replace_conv2d_with_boo_conv(self.model1).to(
                memory_format=torch.channels_last
            )
            _ = model2(x)
            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_forward_b_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )

    def testBackward(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            model = self.model0.to(memory_format=torch.channels_last).train()
            x = torch.ones(
                [10, 2, 16, 16],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ).to(memory_format=torch.channels_last)
            y = model(x)
            y.retain_grad()
            assert y.is_contiguous(memory_format=torch.channels_last)
            loss = y.sum()
            loss.backward()
            # By doing permute calls within the custom op body (instead of outside), the grad_output
            # at the end of the graph (dloss/dy) is actually contiguous rather than in torch.channels_last.
            # It is likely better to let IREE handle the transpose if necessary than to force the layout.
            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_weight_backward_10x16x16x2_nhwc_3x2x2x2_fhwc_nfhw_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_input_backward_10x16x16x2_nhwc_3x2x2x2_fhwc_nfhw_1x1s_0x0p_1x1d_1g",
                func_names,
            )

    def testCompileWithBackwardF32(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones(
                [10, 3, 16, 16],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ).to(memory_format=torch.channels_last)
            model2 = replace_conv2d_with_boo_conv(self.model1).to(
                memory_format=torch.channels_last
            )
            compiled = torch.compile(model2)

            y = compiled(x)
            loss = y.sum()

            loss.backward()

            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_forward_b_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_input_backward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_input_backward_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_weight_backward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_weight_backward_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU to test.")
    def testCompileWithBackwardAMPGPU(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir)
            x = torch.ones(
                [10, 3, 16, 16],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ).to(memory_format=torch.channels_last)
            model2 = replace_conv2d_with_boo_conv(self.model1).to(
                memory_format=torch.channels_last
            )
            compiled = torch.compile(model2)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = compiled(x)
                loss = y.sum()

            loss.backward()

            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_bfloat16_forward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_bfloat16_forward_b_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_bfloat16_input_backward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_bfloat16_input_backward_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_bfloat16_weight_backward_10x16x16x3_nhwc_2x3x3x3_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_bfloat16_weight_backward_10x14x14x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )


if __name__ == "__main__":
    unittest.main()
