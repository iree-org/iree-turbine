import unittest
import tempfile
from pathlib import Path

import torch

from iree.turbine.kernel.boo.modeling import BooConv2d, replace_conv2d_with_boo_conv
from iree.turbine.kernel.boo.conv_exports import set_boo_cache


class BooConv2dTest(unittest.TestCase):
    def setUp(self):
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
            set_boo_cache(cache_dir)
            x = torch.ones([10, 2, 16, 16], device=self.device, dtype=torch.float32)
            _ = self.model0(x)
            self.assertIn(
                "conv_2d_float32_forward_10x2x16x16_nchw_3x2x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                [i.name for i in cache_dir.glob("*")],
            )

    def testReplacement(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_boo_cache(cache_dir)
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
            set_boo_cache(cache_dir)
            x = torch.ones([10, 2, 16, 16], device=self.device, dtype=torch.float32).to(
                memory_format=torch.channels_last
            )
            model = self.model0.to(memory_format=torch.channels_last)
            _ = model(x)
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                [i.name for i in cache_dir.glob("*")],
            )

    def testReplacementChannelsLast(self):
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_boo_cache(cache_dir)
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
            set_boo_cache(cache_dir)
            model = self.model0.to(memory_format=torch.channels_last).train()
            x = torch.ones(
                [10, 2, 16, 16],
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ).to(memory_format=torch.channels_last)
            y = model(x)
            loss = y.sum()
            loss.backward()
            func_names = [i.name for i in cache_dir.glob("*")]
            self.assertIn(
                "conv_2d_float32_forward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_input_backward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )
            self.assertIn(
                "conv_2d_float32_weight_backward_10x16x16x2_nhwc_3x2x2x2_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
                func_names,
            )


if __name__ == "__main__":
    unittest.main()
