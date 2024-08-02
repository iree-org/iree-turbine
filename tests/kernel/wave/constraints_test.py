import logging
import pytest
import unittest
from sympy import ceiling
from shark_turbine.kernel.lang import sym
from shark_turbine.kernel.wave.constraints import (
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

        assert get_grid_shape(constraints) == [M // BLOCK_M, N // BLOCK_N]

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

        assert constraints[0].iterations() == ceiling(M / BLOCK_M)
        assert constraints[1].iterations() == ceiling(N / BLOCK_N)
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
