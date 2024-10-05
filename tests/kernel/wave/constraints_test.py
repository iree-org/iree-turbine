# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import unittest
from sympy import ceiling
from iree.turbine.kernel.lang import sym
from iree.turbine.kernel.wave.constraints import (
    WorkgroupConstraint,
    get_grid_shape,
    TilingConstraint,
    WaveConstraint,
)

M = sym.M
N = sym.N
BLOCK_N = sym.BLOCK_N
BLOCK_M = sym.BLOCK_K
I = sym.I


class ConstraintsTest(unittest.TestCase):
    def testWorkgroupConstraint(self):
        constraints: list[WorkgroupConstraint] = [WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints.append(WorkgroupConstraint(N, BLOCK_N, 1))

        assert get_grid_shape(constraints) == [
            ceiling(M / BLOCK_M),
            ceiling(N / BLOCK_N),
        ]

        # Checking multiple constraints in the same dimension not supported
        constraints += [WorkgroupConstraint(N, BLOCK_N, 1)]
        with pytest.raises(
            ValueError,
            match="Multiple constraints in the same workgroup dimension are currently not supported.",
        ):
            get_grid_shape(constraints)

        # Checking invalid workgroup dimension
        with pytest.raises(
            ValueError,
            match="Invalid workgroup dimension. Expected 0, 1 or 2.",
        ):
            WorkgroupConstraint(N, BLOCK_N, 3).apply()

    def testTilingConstraint(self):
        constraints: list[TilingConstraint] = [TilingConstraint(M, BLOCK_M)]
        constraints.append(TilingConstraint(N, BLOCK_N, I))

        assert constraints[0].count == ceiling(M / BLOCK_M)
        assert constraints[1].count == ceiling(N / BLOCK_N)
        assert constraints[1].apply().start == I * BLOCK_N

        with pytest.raises(
            ValueError,
            match="Index is being computed without setting induction variable",
        ):
            constraints[0].apply()

    def testWaveConstraint(self):
        constraints: list[WaveConstraint] = [WaveConstraint(M, BLOCK_M, I)]
        constraints.append(WaveConstraint(N, BLOCK_N))

        assert constraints[0].apply().start == I * BLOCK_M

        with pytest.raises(
            ValueError,
            match="Index is being computed without setting wave id",
        ):
            constraints[1].apply()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
