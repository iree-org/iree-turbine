import logging
import unittest
from shark_turbine.kernel.lang import sym
from shark_turbine.kernel.wave.utils import delinearize_index
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
