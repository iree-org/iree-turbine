# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging
import unittest

from iree.compiler.ir import (
    Context,
    Operation,
)

from shark_turbine.transforms import rewriter
from shark_turbine.transforms.quantization import mm_group_quant

MM_F32_TO_INT4_CONTENTS = (
    Path(__file__).resolve().parent / "mm_f32_to_int4.mlir"
).read_text()


class Int4Quant(unittest.TestCase):
    def setUp(self):
        self.MM_F32_TO_INT4_CONTENTS = (
            Path(__file__).resolve().parent / "mm_f32_to_int4.mlir"
        ).read_text()

    def testBasic(self):
        with Context() as context:
            module_op = Operation.parse(self.MM_F32_TO_INT4_CONTENTS)
            mm_group_quant.MMGroupQuantRewriterPass(module_op).run()
            module_asm = str(module_op)
            print(module_asm)
            self.assertNotIn("torch.aten.mm", module_asm)
            self.assertNotIn(
                "@_params.model.layers.0.self_attn.q_proj.weight ", module_asm
            )
            self.assertIn("linalg.generic", module_asm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
