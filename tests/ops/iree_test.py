# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch.nn as nn

import shark_turbine.aot as aot
import shark_turbine.ops as ops


# See runtime/op_reg/kernel_aot_test.py for additional tests of the trace
# op.
class TraceTensorTest(unittest.TestCase):
    def testEager(self):
        t = torch.randn(3, 4)
        ops.iree.trace_tensor("TEST", t)

    def testAOT(self):
        class MyModule(nn.Module):
            def forward(self, x):
                ops.iree.trace_tensor("TEST", x)
                return x + 1

        cm = aot.export(MyModule(), args=(torch.empty(9, 8),))
        asm = str(cm.mlir_module)
        self.assertIn('flow.tensor.trace "TEST" =', asm)


class TransferToLogicalDeviceTest(unittest.TestCase):
    def testEager(self):
        t1 = torch.randn(3, 4)
        t2 = ops.iree.transfer_to_logical_device("1", t1)
        self.assertIs(t1, t2)

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
