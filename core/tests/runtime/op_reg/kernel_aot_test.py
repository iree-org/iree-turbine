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

from shark_turbine.transforms.general.custom_op_expansion import ExpandCustomOpsPass


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(8, 8, bias=True)
        self.layer1 = nn.Linear(8, 4, bias=True)
        self.layer2 = nn.Linear(4, 2, bias=True)
        self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        x = torch.sigmoid(x)
        ops.iree.trace_tensor("LAYER0", x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        ops.iree.trace_tensor("LAYER1", x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        ops.iree.trace_tensor("LAYER2", x)
        x = self.layer3(x)
        ops.iree.trace_tensor("LAYER3", x)
        return x


class KernelRegTest(unittest.TestCase):
    def testTrace(self):
        mlp = MLP()
        prog = aot.export(mlp, torch.empty(97, 8, dtype=torch.float32))

        p = ExpandCustomOpsPass(prog.mlir_module)
        p.run()

        print("CUSTOM OP CONVERTED:")
        module_asm = str(prog.mlir_module)
        print(module_asm)
        self.assertIn('flow.tensor.trace "LAYER0"', module_asm)
        self.assertIn('flow.tensor.trace "LAYER1"', module_asm)
        self.assertIn('flow.tensor.trace "LAYER3"', module_asm)

    def testEager(self):
        mlp = MLP()
        mlp.forward(torch.empty(97, 8, dtype=torch.float32))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
