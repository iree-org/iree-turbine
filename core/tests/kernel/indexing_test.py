import re
import unittest

import torch

from shark_turbine.kernel._support.indexing import *

M = sym.M
N = sym.N
K = sym.K


class TestTypes(unittest.TestCase):
    def testGridRepr(self):
        self.assertEqual("Grid", repr(Grid))
        self.assertEqual("Grid[M]", repr(Grid[M]))
        self.assertEqual("Grid[M]", repr(Grid["M"]))
        self.assertEqual("Grid[M, N]", repr(Grid[sym.M, sym.N]))
        self.assertEqual("Grid[M, M/2]", repr(Grid[M, M / 2]))

    def testGridAttrs(self):
        T = Grid[M, N]
        self.assertIs(T.symbolic_shape[0], M)
        self.assertIs(T.symbolic_shape[1], N)
        self.assertEqual(2, T.rank)

    def testShapedGridInstance(self):
        G = Grid[M, N, K]
        with IndexingContext() as idxc:
            idxc.bind_constant(M, 1)
            idxc.bind_constant(N, 2)
            idxc.bind_constant(K, 3)
            idxc.finalize()
            g = G()
        self.assertEqual(3, len(g))
        self.assertEqual(1, g[0])
        self.assertEqual([1, 2, 3], list(g))
        self.assertEqual(3, g.rank)

    def testKernelBufferRepr(self):
        self.assertEqual("KernelBuffer", repr(KernelBuffer))
        self.assertEqual("KernelBuffer[M]", repr(KernelBuffer[sym.M]))
        self.assertEqual("KernelBuffer[M, N]", repr(KernelBuffer[sym.M, sym.N]))
        self.assertEqual("KernelBuffer[M, N]", repr(KernelBuffer["M", "N"]))
        self.assertEqual("KernelBuffer[M, M/2]", repr(KernelBuffer[M, M / 2]))

    def testKernelBufferAttrs(self):
        T = KernelBuffer[M, N]
        self.assertIs(T.symbolic_shape[0], M)
        self.assertIs(T.symbolic_shape[1], N)
        self.assertEqual(2, T.rank)

    def testKernelBufferGenericInstance(self):
        kb = KernelBuffer(torch.empty((3, 4)))
        self.assertEqual(2, kb.rank)

    def testKernelBufferInstance(self):
        T1 = KernelBuffer[M]
        with self.assertRaisesRegex(ValueError, "mismatched symbolic rank"):
            T1(torch.empty((3, 4)))
        kb = T1(torch.empty((3,)))
        self.assertEqual(1, kb.rank)
        self.assertEqual((M,), kb.symbolic_shape)

    def testUsageAndElementTypeInstance(self):
        T = InputBuffer[M].of(torch.float16)
        self.assertEqual("InputBuffer[M].of(torch.float16)", repr(T))


class ContextTest(unittest.TestCase):
    def testConstant(self):
        c = IndexingContext()
        c.bind_constant(M, 4)
        c.finalize()

    def testConstantConflict(self):
        c = IndexingContext()
        c.bind_constant(M, 4)
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Attempt to bind symbol M=5 conflicts with previous 4"),
        ):
            c.bind_constant(M, 5)

    def testKernelBuffers(self):
        c = IndexingContext()
        kb1 = KernelBuffer[M, N]
        c.bind_shaped(object(), kb1, (1, 2))
        c.finalize()

    def testDimConflict(self):
        c = IndexingContext()
        kb1 = KernelBuffer[M, M]
        c.bind_shaped(object(), kb1, (1, 2))
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "KernelBuffer[M, M] attempt to bind dim M=2 conflicts with previous 1"
            ),
        ):
            c.finalize()

    def testDimExprRequiredEquation(self):
        c = IndexingContext()
        inst = object()
        kb1 = KernelBuffer[M, M / 2]
        c.bind_shaped(inst, kb1, (4, None))
        c.finalize()
        self.assertEqual(c.eval_static_dim(inst, kb1, 0), 4)
        self.assertEqual(c.eval_static_dim(inst, kb1, 1), 2)

    def testDimExprRequiredEquationNotSatisfied(self):
        c = IndexingContext()
        kb1 = KernelBuffer[M, N]
        c.bind_shaped(object(), kb1, (4, None))
        with self.assertRaisesRegex(
            ValueError,
            re.escape("KernelBuffer[M, N][1]=N did not resolve to a known value"),
        ):
            c.finalize()

    def testDimExprOptionalDynamicDim(self):
        c = IndexingContext()
        inst = object()
        kb1 = KernelBuffer[M, N]
        c.bind_shaped(inst, kb1, (4, c.next_dyn_dim()))
        c.finalize()
        self.assertEqual(c.dyn_dims[0], c.eval_dim(inst, kb1, 1))

    def testDynamicDimStaticInfoSufficient(self):
        c = IndexingContext()
        inst = object()
        kb1 = KernelBuffer[M, M * 4]
        c.bind_shaped(inst, kb1, (4, c.next_dyn_dim()))
        c.finalize()
        self.assertEqual(16, c.eval_static_dim(inst, kb1, 1))

    def testDimExpressionBackedDynamicDimInferenceMismatch(self):
        c = IndexingContext()
        kb1 = KernelBuffer[M, M / 2]
        c.bind_shaped(object(), kb1, (4, 3))
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "KernelBuffer[M, M/2][1]=2 was initialized with a mismatched runtime value of 3"
            ),
        ):
            c.finalize()

    def testDependentDynamicDims(self):
        c = IndexingContext()
        inst = object()
        kb1 = KernelBuffer[M, M * 4]
        c.bind_shaped(inst, kb1, (c.next_dyn_dim(), c.next_dyn_dim()))
        c.finalize()
        self.assertEqual(c.dyn_dims[0], c.eval_dim(inst, kb1, 0))
        self.assertEqual(c.dyn_dims[0] * 4, c.eval_dim(inst, kb1, 1))


class SymIndexTest(unittest.TestCase):
    def testUnbacked(self):
        idxc = IndexingContext()
        i = SymIndex(idxc)
        self.assertEqual("UnbackedSymIndex", repr(type(i)))

    def testEqual(self):
        idxc = IndexingContext()
        idxc.bind_constant(M, 30)
        idxc.finalize()

        t0 = backed_sym_index_type(EqualRelation(M))
        self.assertEqual("SymIndex==M", repr(t0))
        i0 = t0(idxc)

        t1 = backed_sym_index_type(EqualRelation(M + 1))
        self.assertEqual("SymIndex==(M + 1)", repr(t1))
        i1 = t1(idxc)

    def testBounded(self):
        idxc = IndexingContext()
        idxc.bind_constant(M, 30)
        idxc.finalize()

        t = backed_sym_index_type(BoundedRelation(M, M + 1))
        self.assertEqual("SymIndex∈[M, M + 1]", repr(t))
        i = t(idxc)

        t = backed_sym_index_type(
            BoundedRelation(M, M + 1, lower_inclusive=False, upper_inclusive=False)
        )
        self.assertEqual("SymIndex∈(M, M + 1)", repr(t))

        t = backed_sym_index_type(BoundedRelation(0, M, upper_inclusive=False))
        self.assertEqual("SymIndex∈[0, M)", repr(t))


if __name__ == "__main__":
    unittest.main()
