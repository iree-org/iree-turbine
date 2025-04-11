import unittest
import pytest
import tempfile

from pathlib import Path

import torch

from iree.turbine.kernel.boo.conv_exports.launch import set_boo_cache
from iree.turbine.kernel.boo.ops import boo_conv


class BooConvTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
