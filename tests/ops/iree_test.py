# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import tempfile
import torch
import torch.nn as nn
import numpy as np
import os

import iree.turbine.aot as aot
import iree.turbine.ops as ops
import iree.turbine.support.debugging as debugging


# See runtime/op_reg/kernel_aot_test.py for additional tests of the trace
# op.
class TraceTensorTest(unittest.TestCase):
    def testEager(self):
        t = torch.randn(3, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            stashed_runtime_trace_dir = debugging.flags.runtime_trace_dir
            debugging.flags.runtime_trace_dir = tmp_dir

            ops.iree.trace_tensor("TEST", t)
            recorded_tensor = np.load(os.path.join(tmp_dir, "TEST.npy"))
            np.testing.assert_equal(recorded_tensor, t)

            # recover the original so we don't influence other tests.
            debugging.flags.runtime_trace_dir = stashed_runtime_trace_dir

    def testAOT(self):
        class MyModule(nn.Module):
            def forward(self, x: torch.Tensor):
                ops.iree.trace_tensor("TEST", x)
                return x + 1

        cm = aot.export(MyModule(), args=(torch.empty(9, 8),))
        asm = str(cm.mlir_module)
        self.assertIn('flow.tensor.trace "TEST" =', asm)

    def testAOT_WithDynamicDims(self):
        class MyModule(nn.Module):
            def forward(self, x: torch.Tensor):
                ops.iree.trace_tensor("TEST", x)
                return x + 1

        d0 = torch.export.Dim("d0")
        args = (torch.empty(9, 8),)
        dynamic_shapes = {"x": {0: d0}}
        cm = aot.export(MyModule(), args=args, dynamic_shapes=dynamic_shapes)
        asm = str(cm.mlir_module)
        self.assertIn('flow.tensor.trace "TEST" =', asm)

    def testAOT_DoesNotGetRemovedWhenArgIsUnused(self):
        class MyModule(nn.Module):
            def forward(self, x: torch.Tensor):
                y = x.clone()
                ops.iree.trace_tensor("TEST", y)
                return x + 1

        cm = aot.export(MyModule(), args=(torch.empty(9, 8),))
        asm = str(cm.mlir_module)
        self.assertIn('flow.tensor.trace "TEST" =', asm)


class TransferToLogicalDeviceTest(unittest.TestCase):
    def testEager(self):
        t1 = torch.randn(3, 4)
        t2 = ops.iree.transfer_to_logical_device("1", t1)
        assert torch.all(t1 == t2)

    def testAOT(self):
        class MyModule(nn.Module):
            def forward(self, x):
                y = x + 1
                z = ops.iree.transfer_to_logical_device("1", y)
                return z + 1

        cm = aot.export(MyModule(), args=(torch.empty(9, 8),))
        asm = str(cm.mlir_module)
        self.assertRegex(
            asm, "flow.tensor.transfer %.+ to #hal.device.promise<@__device.1>"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
