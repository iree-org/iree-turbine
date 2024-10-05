# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl

from iree.turbine.kernel.compiler import (
    builder,
    dispatch_codegen,
    kernel_codegen,
    vector_codegen,
)
from iree.turbine.kernel._support import (
    indexing,
)


M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    def testEmptyStreamExecutable(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.InputBuffer[M, K, tkl.f32],
            output: tk.lang.OutputBuffer[M, K, tkl.f32],
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = tkl.exp2(input_row - tkl.max(input_row))
            output_row = numerator / tkl.sum(numerator)
            output[row_index, :] = output_row

        input = torch.randn(128, 64)
        output = torch.zeros(128, 64)
        with tk.gen.TestLaunchContext():
            softmax_kernel(input, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
