# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest


class TopLevelPackageTest(unittest.TestCase):
    def testIreeTurbineRedirect(self):
        # We have a temporary redirect of the top-level API to the
        # iree.turbine namespace.
        from iree.turbine import aot, dynamo, kernel, ops, runtime


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
