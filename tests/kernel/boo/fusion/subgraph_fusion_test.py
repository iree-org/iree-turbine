# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import tempfile

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


if __name__ == "__main__":
    unittest.main()
