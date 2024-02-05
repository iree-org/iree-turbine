import logging
import unittest

import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

from shark_turbine.kernel.compiler import (
    builder,
    dispatch_codegen,
    kernel_codegen,
    vector_codegen,
)
from shark_turbine.kernel._support import (
    indexing,
)


M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    def testEmptyStreamExecutable(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.InputBuffer[M, K], output: tk.lang.OutputBuffer[M, K]
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = tkl.exp2(input_row - tkl.max(input_row))
            output_row = numerator / tkl.sum(numerator)
            output[row_index, :] = output_row

        trace = softmax_kernel._trace
        print(trace.region_graph)
        mb = builder.ModuleBuilder()
        with indexing.IndexingContext() as idxc:
            idxc.bind_shaped(0, tk.lang.InputBuffer[M, K], (128, 64))
            idxc.bind_shaped(1, tk.lang.OutputBuffer[M, K], (128, 64))
            idxc.finalize()

            sig = kernel_codegen.KernelSignature()
            sig.add_from_graph_placeholders(trace.get_root_graph())
            sig.add_grid(softmax_kernel.grid_type)

            try:
                exe = dispatch_codegen.StreamExecutable(mb)
                dispatch_entrypoint = exe.define_entrypoint("dispatch", sig)
                emitter = vector_codegen.ThreadEmitter(dispatch_entrypoint, trace)
                emitter.emit()
                emitter.finish()
            finally:
                print(mb.module_op.get_asm())
            mb.module_op.verify()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
