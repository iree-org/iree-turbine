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

M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    # This test is using the compiler "the hard way" until we have all of the
    # API layering in place.
    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.OutputBuffer[M]):
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

        trace = iota_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_constant(M, 17)
            idxc.finalize()
            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(iota_kernel.grid_type)
            print(sig)
            bound_sig, func_op = kernel_codegen.FunctionalKernelSignature.create(
                sig, mb
            )
            try:
                emitter = vector_codegen.ThreadEmitter(bound_sig, trace)
                emitter.emit()
                emitter.finish()
            finally:
                print(mb.module_op.get_asm())
            mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
