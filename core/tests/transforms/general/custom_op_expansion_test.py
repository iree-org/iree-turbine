# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import torch
import unittest

from shark_turbine.transforms.general.custom_op_expansion import ExpandCustomOpsPass
from shark_turbine.runtime.op_reg import (
    def_library,
    CustomOp,
    KernelBuilder,
    KernelSelection,
)

from shark_turbine.support.ir_imports import (
    Context,
    Module,
)


class PassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = def_library("expand_custom_op_pass_test")
        CustomOp.register(library=cls.lib)(IdentityOp)
        CustomOp.register(library=cls.lib)(PrintStringAttrOp)
        CustomOp.register(library=cls.lib)(IntArgOp)

    def testTensorArgReturn(self):
        m = self.run_test_case("custom_op_simple.mlir")
        m_asm = str(m)
        self.assertNotIn("torch.operator", m_asm)
        self.assertIn(
            "%0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[97,8],f32> -> tensor<97x8xf32>",
            m_asm,
        )
        self.assertIn(
            "%1 = torch_c.from_builtin_tensor %0 : tensor<97x8xf32> -> !torch.vtensor<[97,8],f32>",
            m_asm,
        )
        print(m_asm)

    def testStringAttrArg(self):
        global _TEST_STRING_ATTR
        _TEST_STRING_ATTR = ""
        m = self.run_test_case("custom_op_string_attr.mlir")
        m_asm = str(m)
        self.assertEqual(_TEST_STRING_ATTR, "TEST_VALUE")
        self.assertNotIn("torch.operator", m_asm)
        print(m_asm)

    def testIntArg(self):
        global _TEST_STRING_ATTR
        _TEST_STRING_ATTR = ""
        with self.assertRaisesRegex(NotImplementedError, "arg_int"):
            self.run_test_case("custom_op_int_arg.mlir")

    def run_test_case(self, file_name: str):
        p = Path(__file__).resolve().parent / "testdata" / file_name
        contents = p.read_text()
        with Context() as ctx:
            m = Module.parse(contents)
        p = ExpandCustomOpsPass(m.operation)
        p.run()
        print(f"TEST CASE {file_name}:\n{m}")
        m.operation.verify()
        return m


class IdentityOp(CustomOp):
    signature = "identity_tensor(Tensor t) -> Tensor"

    def select(self, ksel: KernelSelection):
        x = ksel.arg_tensor(0)
        ksel.return_tensor(x.t)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        kb.yield_results(kb.arg_bindings[0])


class PrintStringAttrOp(CustomOp):
    signature = "print_string_attr(str key) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        global _TEST_STRING_ATTR
        _TEST_STRING_ATTR = str(ksel.arg_descs[0].v)
        print("CAPTURED STRING ATTR:", _TEST_STRING_ATTR)
        kb.yield_results()


class IntArgOp(CustomOp):
    signature = "int_arg(int t) -> ()"

    def select(self, ksel: KernelSelection):
        x = ksel.arg_int(0)
        ksel.return_int()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        kb.yield_results(kb.arg_bindings[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
