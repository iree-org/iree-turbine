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
            "func.func @foobar(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<3x2xf32>)",
            module_str,
        )
        self.assertIn("return %arg1, %arg0", module_str)

    def testProcToJitArgs(self):
        class ProcArgsModule(CompiledModule):
            def foobar(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                return self.compute(a, b)

            @jittable
            def compute(a, b):
                return a + b

        inst = ProcArgsModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "func.func @foobar(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<3x2xf32>",
            module_str,
        )
        self.assertIn(
            "func.func private @compute(%arg0: tensor<3x2xf32>, %arg1: tensor<1x1xf32>) -> tensor<3x2xf32>",
            module_str,
        )
        self.assertIn(
            "%0 = call @compute(%arg0, %arg1)",
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
        self.assertIn(
            "%0 = call @compute(%arg0, %arg1)",
            module_str,
        )
        self.assertIn(
            "%1 = call @compute$1(%0, %arg0)",
            module_str,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
