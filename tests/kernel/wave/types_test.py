import logging
import pytest
import sympy
import unittest

from shark_turbine.kernel.lang import Memory, Register, sym, f16
from shark_turbine.kernel.lang.wave_types import AddressSpace
from shark_turbine.kernel.lang.kernel_buffer import KernelBufferUsage

M = sym.M
N = sym.N
ADDRESS_SPACE = sym.ADDRESS_SPACE


class MemoryTest(unittest.TestCase):
    def testIndexType(self):
        # Check single dimensional
        v = Memory[M, ADDRESS_SPACE, f16]
        self.assertEqual(v.symbolic_shape, (sym.M,))
        self.assertEqual(v.address_space, sym.ADDRESS_SPACE)
        self.assertEqual(v.dtype, f16)

        # Check multi dimensional
        A = Memory[M, N, ADDRESS_SPACE, f16]
        self.assertEqual(A.symbolic_shape, (sym.M, sym.N))
        self.assertEqual(A.address_space, sym.ADDRESS_SPACE)
        self.assertEqual(A.dtype, f16)

        # Check specifying usage
        B = Memory[M, N, ADDRESS_SPACE, f16, KernelBufferUsage.INPUT]
        self.assertEqual(B.usage, KernelBufferUsage.INPUT)

        # Check constant sizes
        C = Memory[32, 16, ADDRESS_SPACE, f16]
        self.assertEqual(C.symbolic_shape, (sympy.Expr(32), sympy.Expr(16)))

        # Check specific address space
        D = Memory[M, N, AddressSpace.SHARED_MEMORY, f16]
        self.assertEqual(D.address_space, AddressSpace.SHARED_MEMORY)

        with pytest.raises(
            TypeError,
            match="Memory does not support address space register, use Register instead.",
        ):
            Memory[M, N, AddressSpace.REGISTER, f16]

        with pytest.raises(
            TypeError, match="Expected addressSpace to be a AddressSpace, got 1"
        ):
            Memory[M, N, 1, f16]

        with pytest.raises(TypeError, match="Expected dtype to be a DataType"):
            Memory[M, N, ADDRESS_SPACE, KernelBufferUsage.INPUT]

        with pytest.raises(
            TypeError, match="Expected shape to be a tuple of IndexExpr, got ()"
        ):
            Memory[ADDRESS_SPACE, f16, KernelBufferUsage.INPUT]

        with pytest.raises(
            NotImplementedError, match="Memory types are not directly instantiated."
        ):
            Memory()


class RegisterTest(unittest.TestCase):
    def testIndexType(self):
        # Check single dimensional
        v = Register[M, f16]
        self.assertEqual(v.symbolic_shape, (sym.M,))
        self.assertEqual(v.dtype, f16)

        # Check multi dimensional
        A = Register[M, N, f16]
        self.assertEqual(A.symbolic_shape, (sym.M, sym.N))
        self.assertEqual(A.dtype, f16)

        # Check constant sizes
        C = Register[32, 16, f16]
        self.assertEqual(C.symbolic_shape, (sympy.Expr(32), sympy.Expr(16)))

        with pytest.raises(TypeError, match="Expected dtype to be a DataType"):
            Register[M, N]

        # See Register instantiation test in tracing tests


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
