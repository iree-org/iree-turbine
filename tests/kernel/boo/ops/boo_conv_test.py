import os
# enable backward boo kernels for testing
os.environ["BOO_USE_BACKWARD_KERNELS"] = "1"

import unittest
import pytest
import tempfile

from pathlib import Path

import torch

from iree.turbine.kernel.boo.conv_exports.launch import (
    set_boo_cache,
    ConvLaunchableRuntimeCache,
)
from iree.turbine.kernel.boo.ops import boo_conv


class BooConvTest(unittest.TestCase):
    def setUp(self):
        ConvLaunchableRuntimeCache.set_cache_limit(0)

    def testBooConvNonDefault(self):
        with tempfile.TemporaryDirectory() as td:
            set_boo_cache(Path(td))
            device = "cuda:0" if torch.cuda.is_available() else None
            x = torch.ones([2, 16, 16, 3], dtype=torch.float32, device=device)
            w = torch.ones([4, 2, 2, 3], dtype=torch.float32, device=device)
            y = boo_conv(x, w, shared_layout="NHWC", stride=2, dilation=2)
            y_exp = torch.ones_like(y, device=device) * 12.0
            self.assertAlmostEqual(
                torch.abs(y - y_exp).sum().item(),
                0.0,
                msg=f"Expected output to be close to splat 12.0 tensor. Got {y}",
            )

    def testBooConvBackwardDefault(self):
        with tempfile.TemporaryDirectory() as td:
            set_boo_cache(Path(td))
            device = "cuda:0" if torch.cuda.is_available() else None
            x = torch.ones(
                [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=True
            )
            w = torch.ones(
                [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=True
            )
            torch.autograd.gradcheck(boo_conv, (x, w), atol=1e-5, eps=1e-3)

    @pytest.mark.xfail(msg="Currently failing. Unmark when issue #704 is resolved.")
    def testBooConvBackwardsWithBias(self):
        with tempfile.TemporaryDirectory() as td:
            set_boo_cache(Path(td))
            device = "cuda:0" if torch.cuda.is_available() else None
            x = torch.ones(
                [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=True
            )
            w = torch.ones(
                [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=True
            )
            b = torch.ones([1], dtype=torch.float32, device=device, requires_grad=True)
            torch.autograd.gradcheck(boo_conv, (x, w, b), atol=1e-5, eps=1e-3)

    def testBooConvBackwardsAmpContextCPU(self):
        """We expect this to not perform autocasting."""

        device = None
        x = torch.ones(
            [1, 1, 32, 32], dtype=torch.float32, device=device, requires_grad=True
        )
        w = torch.ones(
            [1, 1, 4, 4], dtype=torch.float32, device=device, requires_grad=True
        )

        with tempfile.TemporaryDirectory() as td:
            set_boo_cache(Path(td))

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = boo_conv(x, w)
                loss = y.sum()

            loss.backward()

            items = [x.name for x in Path(td).glob("*/")]
            expected_dtype_str = "float32"
            unexpected_dtype_str = "bfloat16"
            self.assertNotIn(
                f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )

            self.assertEqual(y.dtype, torch.float32)
            self.assertEqual(x.grad.dtype, torch.float32)
            self.assertEqual(w.grad.dtype, torch.float32)
            self.assertEqual(w.dtype, torch.float32)

        with tempfile.TemporaryDirectory() as td_0:
            set_boo_cache(Path(td_0))

            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                y = boo_conv(x, w)
                loss = y.sum()

            loss.backward()

            items = [x.name for x in Path(td_0).glob("*/")]
            expected_dtype_str = "bfloat16"
            unexpected_dtype_str = "float32"
            self.assertNotIn(
                f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            # Make sure we got back the correct original dtypes.
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(x.grad.dtype, torch.float32)
            self.assertEqual(w.grad.dtype, torch.float32)
            self.assertEqual(w.dtype, torch.float32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU to test.")
    def testBooConvBackwardsAmpContextCUDA(self):
        with tempfile.TemporaryDirectory() as td:
            set_boo_cache(Path(td))
            device = "cuda:0"
            x = torch.ones(
                [1, 1, 32, 32], dtype=torch.float32, device=device, requires_grad=True
            )
            w = torch.ones(
                [1, 1, 4, 4], dtype=torch.float32, device=device, requires_grad=True
            )
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = boo_conv(x, w)
                loss = y.sum()

            loss.backward()
            items = [x.name for x in Path(td).glob("*/")]
            expected_dtype_str = "bfloat16"
            unexpected_dtype_str = "float32"
            self.assertNotIn(
                f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            self.assertIn(
                f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g",
                items,
            )
            # Make sure we got back the correct original dtypes.
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(x.grad.dtype, torch.float32)
            self.assertEqual(w.grad.dtype, torch.float32)
            self.assertEqual(w.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
