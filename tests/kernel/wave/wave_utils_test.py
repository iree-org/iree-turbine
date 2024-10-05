# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
from iree.turbine.kernel.lang import sym
from iree.turbine.kernel.wave.utils import delinearize_index
import numpy as np

M = sym.M


class UtilsTest(unittest.TestCase):
    def testDelinearizeIndex(self):
        shape = [5, 4, 3]
        nd_index = delinearize_index(M, shape)
        np_nd_index = np.unravel_index(23, shape)
        assert np.equal([x.subs({M: 23}) for x in nd_index], np_nd_index).all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
