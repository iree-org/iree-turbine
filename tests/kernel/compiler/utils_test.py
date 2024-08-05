import logging
import pytest
import unittest
from shark_turbine.kernel.lang import sym
from shark_turbine.kernel._support.indexing import IndexSymbol, IndexingContext
from shark_turbine.kernel.compiler.utils import strides_from_symbolic_shape


class UtilsTest(unittest.TestCase):
    def testStrideComputation(self):
        symbolic_shape = [sym.M, sym.N, sym.K]
        idxc = IndexingContext()
        idxc.bind_constant(sym.M, 64)
        idxc.bind_constant(sym.N, 128)
        idxc.bind_constant(sym.K, 256)
        idxc.finalize()
        strides = strides_from_symbolic_shape(idxc, symbolic_shape)
        assert strides == [128 * 256, 256, 1]

    def testSingleStrideComputation(self):
        symbolic_shape = [sym.M]
        idxc = IndexingContext()
        idxc.bind_constant(sym.M, 64)
        idxc.finalize()
        strides = strides_from_symbolic_shape(idxc, symbolic_shape)
        assert strides == [1]

    def testInvalidSymbolicShape(self):
        symbolic_shape = None
        idxc = IndexingContext()
        idxc.finalize()
        strides = strides_from_symbolic_shape(idxc, symbolic_shape)
        assert strides is None

    def testDynamicSymbolStrideComputation(self):
        symbolic_shape = [sym.M, sym.N, sym.K]
        idxc = IndexingContext()
        idxc.bind_constant(sym.M, 64)
        idxc.bind_constant(sym.K, 256)
        idxc.finalize()
        strides = strides_from_symbolic_shape(idxc, symbolic_shape)
        assert strides is None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
