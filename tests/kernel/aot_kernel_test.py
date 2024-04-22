# Copyright 2024 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import unittest

import torch
from shark_turbine.aot import export
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl


def export_softmax_kernel():
    M = tkl.sym.M
    N = tkl.sym.K

    @tk.gen.kernel(M)
    def softmax(
        input: tkl.InputBuffer[M, N, tkl.f16], output: tkl.OutputBuffer[M, N, tkl.f16]
    ):
        row_index = tkl.program_id(0)
        row = tkl.load(input, (row_index, 0), (1, N))
        row_minus_max = row - tkl.max(row)
        numerator = tkl.exp2(row_minus_max)
        denominator = tkl.sum(numerator)
        softmax_output = numerator / denominator
        tkl.store(output, (row_index, 0), softmax_output)

    class NN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64, dtype=torch.float16)

        def forward(self, x):
            x = self.linear(x)
            x = softmax(x)
            return x

    model = NN()
    a = torch.ones(64, 64, dtype=torch.float16)
    exported = export(model, a)
    return exported


class AotKernelTest(unittest.TestCase):
    def test_unique_naming(self):
        # We test it twice to ensure that local name collisions cannot happen,
        # verifying that each run generates a uniquely named kernel. This is
        # a by-product of the Torch namespace being global and every one of
        # these that we define being a separate incarnation based on the
        # same local function name.
        unique_names = set()
        for _ in range(2):
            exported = export_softmax_kernel()
            exported.print_readable()
            ir_text = str(exported.mlir_module)
            matches = re.findall(
                r"flow.dispatch @(tk_kernel_softmax__([0-9]+))::", ir_text
            )
            self.assertEqual(1, len(matches))
            match = matches[0]
            print("NAME MATCH:", match)
            self.assertNotIn(match, unique_names)
            unique_names.add(match)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
