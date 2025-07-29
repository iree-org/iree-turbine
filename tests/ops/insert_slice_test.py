# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

import iree.turbine.aot as aot
from iree.turbine.ops.insert_slice import insert_slice, generic_insert_slice


class InsertSliceTest(unittest.TestCase):
    def testEager(self):
        src = torch.ones((4, 2))
        dst = torch.zeros((10, 4))
        y = insert_slice(src, dst, [0, 2], [2, 1])
        y_ref = dst
        y_ref[0:8:2, 2:4:1] = src
        diff = torch.max(torch.abs(y - y_ref)).item()
        self.assertAlmostEqual(diff, 0.0)

    def testEagerGeneric(self):
        src = torch.ones((4, 2))
        y = generic_insert_slice(src, [4, 4], [0, 0], [1, 2])
        y_ref = torch.zeros([4, 4])
        y_ref[:, 0:-1:2] = 1.0
        self.assertEqual(torch.max(torch.abs(y - y_ref)).item(), 0.0)

    def testAOT(self):
        class M(torch.nn.Module):
            def __init__(self, offsets, strides):
                super().__init__()
                self.offsets = offsets
                self.strides = strides

            def forward(self, src, dst):
                return insert_slice(src, dst, self.offsets, self.strides)

        e = aot.export(M([0, 2], [2, 1]), args=(torch.empty(4, 2), torch.empty(10, 4)))
        mlir_asm = str(e.mlir_module)
        self.assertIn("tensor.insert_slice", mlir_asm)

    def testAOTGeneric(self):
        class M(torch.nn.Module):
            def __init__(self, sizes, offsets, strides):
                super().__init__()
                self.sizes = sizes
                self.offsets = offsets
                self.strides = strides

            def forward(self, src):
                return generic_insert_slice(src, self.sizes, self.offsets, self.strides)

        e = aot.export(M([4, 4], [0, 0], [1, 2]), args=(torch.empty(4, 2),))
        mlir_asm = str(e.mlir_module)
        self.assertIn("linalg.generic", mlir_asm)
