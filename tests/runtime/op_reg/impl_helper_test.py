# Copyright 2023 Advanced Micro Devices, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

from shark_turbine.ops import _str_format_test_ops


class KernelRegTest(unittest.TestCase):
    def testError(self):
        t = torch.randn(3, 4)
        try:
            _str_format_test_ops.syntax_error(t)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            message = str(e)
            self.assertIn("error:", message)
            self.assertIn("1: THIS IS A SYNTAX ERROR", message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
