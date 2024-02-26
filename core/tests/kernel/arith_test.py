import logging
import unittest

import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

from shark_turbine.kernel.compiler import (
    builder,
    kernel_codegen,
    vector_codegen,
)
from shark_turbine.kernel._support import (
    indexing,
)

M = tkl.sym.M
K = tkl.sym.K


class Test(unittest.TestCase):
    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tkl.OutputBuffer[M, tkl.f32]):
            # Integer types
            for dtype in [
                tkl.bool,
                tkl.i4,
                tkl.i8,
                tkl.i16,
                tkl.i32,
                tkl.i64,
                tkl.index,
            ]:
                a = tkl.constant((17, 37, 19), dtype, 5)
                b = tkl.constant((17, 37, 19), dtype, 10)
                c = tkl.constant((17, 37, 19), dtype, 2)
                c = (a * b) // c
                c = c + a - b

            # Float types
            for dtype in [tkl.f16, tkl.f32, tkl.f64]:
                a = tkl.constant((17, 37, 19), dtype, 5.0)
                b = tkl.constant((17, 37, 19), dtype, 10.0)
                c = tkl.constant((17, 37, 19), dtype, 2.0)
                c = (a * b) / c
                c = c + a - b

        with tk.gen.TestLaunchContext():
            iota_kernel(torch.zeros(17))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
