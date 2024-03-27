# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

import shark_turbine.ops as ops


class KernelRegTest(unittest.TestCase):
    def testTrace(self):
        t = torch.randn(3, 4)
        ops.iree.trace_tensor("TEST", t)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
