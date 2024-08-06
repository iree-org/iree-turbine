# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

from iree.compiler.ir import (
    Context,
)

from shark_turbine.aot import *


class FunctionalizeTests(unittest.TestCase):
    def testImportPhases(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                return self.compute(), self.compute()

            @CompiledModule.jittable
            def compute():
                offset = torch.ones(2, 2)
                t1 = torch.ones(2, 2)
                t1.add_(offset)
                return t1 * t1

        inst = ExportedProcModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("add_", module_str)

    def testDynamicDims(self):
        class ProcArgsModule(CompiledModule):
            def dynamic_dim(self, a=AbstractTensor(None, 2), b=AbstractTensor(None, 1)):
                dim0 = torch.export.Dim("dim0")
                dynamic_shapes = {"arg0_1": {0: dim0}, "arg1_1": {0: dim0}}
                return self.compute(
                    a,
                    b,
                    dynamic_shapes=dynamic_shapes,
                )

            @jittable
            def compute(a, b):
                a.mul_(b)
                return a

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("mul_", module_str)

    def testCallWithStructure(self):
        class ProcArgsModule(CompiledModule):
            def call_with_dicts(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                intermediate = self.compute({"a": a, "b": b})
                return self.compute(intermediate)

            @jittable
            def compute(struct):
                a = struct["a"]
                b = struct["b"]
                a.add_(b)
                return {"a": a, "b": b}

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("add_", module_str)

    def testCallWithArgsKwargs(self):
        class ProcArgsModule(CompiledModule):
            def call_with_kwargs(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                intermediate = self.compute(**{"a": a, "b": b})
                return self.compute(**intermediate)

            @jittable
            def compute(*, a, b):
                a.add_(b)
                return {"a": a, "b": b}

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("add_", module_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
