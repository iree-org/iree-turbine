# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.compiler.ir import (
    Context,
)

from shark_turbine.aot import *


class ArgsTest(unittest.TestCase):
    def testProcArgs(self):
        class ProcArgsModule(CompiledModule):
            def foobar(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                return b, a

        inst = ProcArgsModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "util.func public @foobar$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)",
            module_str,
        )

    def testProcToJitArgs(self):
        class testProcToJitArgs(CompiledModule):
            def foobar(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                return self.compute(a, b)

            @jittable
            def compute(a, b):
                return a + b

        inst = testProcToJitArgs(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "linalg.generic",
            module_str,
        )

    def testProcToJitArgsMultiCall(self):
        class ProcArgsModule(CompiledModule):
            def foobar(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                x = self.compute(a, b)
                y = self.compute(x, a)
                return y

            @jittable
            def compute(a, b):
                return a + b

        inst = ProcArgsModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertEqual(
            2,
            module_str.count("linalg.generic"),
            msg=f"Did not find two linalg.generics in module: module_str",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
