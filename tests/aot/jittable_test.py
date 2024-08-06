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


class JittableTests(unittest.TestCase):
    def testImportPhases(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                return self.compute(), self.compute()

            @CompiledModule.jittable
            def compute():
                t1 = torch.ones(2, 2)
                t2 = t1 + t1
                return t2 * t2

        inst = ExportedProcModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        # Functions should still be on torch types.
        self.assertIn(
            "func private @compute() -> !torch.vtensor<[2,2],f32>", module_str
        )
        CompiledModule.run_import(inst, import_to="full")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertNotIn("!torch.vtensor", module_str)

    def testCallWithStructure(self):
        class ProcArgsModule(CompiledModule):
            def call_with_dicts(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                intermediate = self.compute({"a": a, "b": b})
                return self.compute(intermediate)

            @jittable
            def compute(struct):
                a = struct["a"]
                b = struct["b"]
                result = a + b
                return {"a": result, "b": b}

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testCallWithArgsKwargs(self):
        class ProcArgsModule(CompiledModule):
            def call_with_kwargs(self, a=AbstractTensor(3, 2), b=AbstractTensor(1, 1)):
                intermediate = self.compute(**{"a": a, "b": b})
                return self.compute(**intermediate)

            @jittable
            def compute(*, a, b):
                result = a + b
                return {"a": result, "b": b}

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testDynamicDims(self):
        class DynamicDimsModule(CompiledModule):
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
                return a * b

        inst = DynamicDimsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testIntTensors(self):
        class ProcArgsModule(CompiledModule):
            def dynamic_dim(
                self,
                a=AbstractTensor(2, 2, dtype=torch.int64),
                b=AbstractTensor(1, 1, dtype=torch.int64),
            ):
                return self.compute(a, b)

            @jittable
            def compute(a, b):
                return a * b

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testIrImmediateTensorAsInputToDynamicDims(self):
        class ProcArgsModule(CompiledModule):
            def dynamic_dim(self, x=AbstractIndex):
                a = IREE.tensor_empty(x, 4)
                b = IREE.tensor_empty(x, 4)
                dim0 = torch.export.Dim("dim0")
                dynamic_shapes = {"arg0_1": {0: dim0}, "arg1_1": {0: dim0}}
                return self.compute(a, b, dynamic_shapes=dynamic_shapes)

            @jittable
            def compute(a, b):
                return a * b

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)

    def testImplicitTensorsImportedAndDeduped(self):
        implicit = torch.tensor([1, 2, 3], dtype=torch.int32)

        class ProcArgsModule(CompiledModule):
            def implicit(
                self,
                a=AbstractTensor(3, dtype=torch.int32),
            ):
                return self.compute(a), self.compute(a)

            @jittable
            def compute(a):
                return a * implicit

        inst = ProcArgsModule(context=Context(), import_to=None)
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        # This is testing machinery which ensures that not only are
        # implicit captured tensors emitted as resources, but that
        # multiple subsequent references to the same tensor is
        # only captured once.
        resource_string = (
            r'''torch_tensor_3_torch.int32: "0x04000000010000000200000003000000"'''
        )
        self.assertIn(resource_string, module_str)
        self.assertEqual(
            1,
            module_str.count(resource_string),
            f"Expected to find exactly one of '{resource_string}' in '{module_str}'",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
