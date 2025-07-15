# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import torch
import iree.turbine.kernel.boo.ops as boo_ops
import iree.turbine.aot as aot


class SimpleAOTTest(unittest.TestCase):
    def testAOTLayoutCustomizable(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        class SampleModule(torch.nn.Module):
            def forward(self, x, w):
                return (
                    boo_ops.boo_layout_customizable_convolution(
                        x, w, None, [1, 1], [0, 0], [1, 1], 1, "NHWC", "NHWC", "NHWC"
                    )
                    * 0.1
                )

        N = 2
        C = 32
        H = 16
        W = 16
        k = 1
        f = 2
        x = torch.randn([N, H, W, C], device=device)
        w = torch.randn([f, k, k, C], device=device)
        e = aot.export(SampleModule(), args=(x, w))
        e.mlir_module.verify()
        self.assertIn(
            "call @conv_2d_float32_forward_2x16x16x32_nhwc_2x1x1x32_fhwc_nhwf_1x1s_0x0p_1x1d_1g",
            str(e.mlir_module),
        )


if __name__ == "__main__":
    unittest.main()
