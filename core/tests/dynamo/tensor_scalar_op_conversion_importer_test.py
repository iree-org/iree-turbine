# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from testutils import *


class TensorScalarOpConversionImportModule(unittest.TestCase):
    def setUp(self):
        self.t = torch.randn(2, 2)

    def add(self, x):
        return x + 8.2

    def sub(self, x):
        return x - 1.6

    def mul(self, x):
        return x * 3.2

    def div(self, x):
        return x / 2.1

    def floor_div(self, x):
        return x // 4.2

    def testAdd(self):
        opt_torch_scalar_convert = torch.compile(self.add, backend=create_backend())
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.add(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def testSub(self):
        opt_torch_scalar_convert = torch.compile(self.sub, backend=create_backend())
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.sub(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def testMul(self):
        opt_torch_scalar_convert = torch.compile(self.mul, backend=create_backend())
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.mul(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def testDiv(self):
        opt_torch_scalar_convert = torch.compile(self.div, backend=create_backend())
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")

    def testFloorDiv(self):
        """
        This op isn't successfully created by IREE due to partial implementation of floor_div op in torch-mlir
        However, the importer works successfully.
        """
        opt_torch_scalar_convert = torch.compile(
            self.floor_div, backend=create_backend()
        )
        result = opt_torch_scalar_convert(self.t)
        expected_result = self.floor_div(self.t)
        self.assertTrue(torch.allclose(result, expected_result), "broken")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
