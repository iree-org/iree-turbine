import logging
import pytest
import unittest
from shark_turbine.kernel.lang import sym
from shark_turbine.kernel.wave.constraints import WorkgroupConstraint, get_grid_shape

M = sym.M
N = sym.N
BLOCK_N = sym.BLOCK_N
BLOCK_M = sym.BLOCK_K


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
            match="Invalid workgroup dimension. Expected 0 or 1.",
        ):
            WorkgroupConstraint(N, BLOCK_N, 2).apply()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
