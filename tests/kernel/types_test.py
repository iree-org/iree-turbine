# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.turbine.kernel.lang import (
    Index,
)


class IndexTypeTest(unittest.TestCase):
    def testIndexType(self):
        i = Index(5)
        j = Index(-6)
        self.assertIndexEqual(-1, i + j)
        self.assertIndexEqual(11, i - j)
        self.assertIndexEqual(-30, i * j)
        self.assertIndexEqual(-1, i // j)
        self.assertIndexEqual(2, Index(20) % Index(18))
        self.assertIndexEqual(16, Index(4) ** Index(2))
        self.assertIndexEqual(1, pow(Index(6), Index(8), Index(5)))
        self.assertIndexEqual(-6, +j)
        self.assertIndexEqual(6, -j)

    def assertIndexEqual(self, expected: int, actual):
        self.assertEqual(expected, actual)
        self.assertIsInstance(actual, Index)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
