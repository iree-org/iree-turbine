# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import unittest

import logging
from iree.turbine.ops.mlir_kernel import *
import iree.turbine.aot as aot

N = DynDim.N
M = StaticDim.M

S = Dtype.S
I64 = Dtype.I64


@mlir_kernel(
    library="test",
    inputs=(MLIRTensor[N, M, S], MLIRTensor[N, I64]),
    results=(MLIRTensor[N, M, S],),
)
def gather(source, indices, result=None):
    mlir = """
    module {
    util.func @{{kernel_name}}(%source: !source, %indices: !indices) -> !result {
      %c0 = arith.constant 0 : index
      %n_dim = tensor.dim %source, %c0 : !source
      %empty = tensor.empty(%n_dim) : !result
      %result = linalg.generic {
        indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%indices : !indices)
        outs(%empty : !result) {
        ^bb0(%in: !indices_dtype, %o: !source_dtype):
          %n = arith.index_cast %in : !indices_dtype to index
          %m = linalg.index 1 : index
          %extracted = tensor.extract %source[%n, %m] : !source
          linalg.yield %extracted : !source_dtype
        } -> !result
        util.return %result : !result
    }
    }
    """
    return MLIRSpec(mlir)


class mlir_kernel_test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def testEager(self):
        source = torch.randn([64, 32]).to(torch.float16)
        indices = torch.tensor([3, 7, 54])
        out = gather(source, indices)
        torch.testing.assert_close(source[3], out[0])
        torch.testing.assert_close(source[7], out[1])
        torch.testing.assert_close(source[54], out[2])

    def testAOT(self):
        class MyModule(torch.nn.Module):
            def forward(self, source: torch.Tensor, indices: torch.Tensor):
                return gather(source, indices)

        source = torch.randn([64, 32]).to(torch.float16)
        indices = torch.tensor([3, 7, 54])
        cm = aot.export(MyModule(), args=(source, indices))
        asm = str(cm.mlir_module)
        self.assertRegex(asm, "util.func public @gather_N_M_32_f16_N_i64_N_M_32_f16")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
