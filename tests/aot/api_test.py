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

import torch
import torch.nn as nn


class GeneralAPI(unittest.TestCase):
    def testTypedefs(self):
        self.assertEqual(
            "AbstractTensor(3, 2, dtype=torch.float16)",
            repr(AbstractTensor(3, 2, dtype=torch.float16)),
        )


class CompiledModuleAPI(unittest.TestCase):
    def testBasic(self):
        class BasicModule(CompiledModule):
            ...

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("module @basic", module_str)

    def testExplicitName(self):
        class BasicModule(CompiledModule, export_name="explicit"):
            ...

        inst = BasicModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("module @explicit", module_str)

    def testJittableFunc(self):
        class BasicModule(CompiledModule):
            @CompiledModule.jittable
            def mul(x, y):
                return x * y

        inst = BasicModule(context=Context())
        self.assertIsInstance(inst.mul, builtins.jittable)

    def testBareJittableFunc(self):
        class BasicModule(CompiledModule):
            @jittable
            def mul(x, y):
                return x * y

        inst = BasicModule(context=Context())
        self.assertIsInstance(inst.mul, builtins.jittable)

    def testExportedProc(self):
        class ExportedProcModule(CompiledModule):
            def foobar(self):
                ...

        inst = ExportedProcModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)


class ExportAPI(unittest.TestCase):
    def testStaticNNModule(self):
        mdl = SimpleParams()
        exported = export(mdl, args=(torch.empty([128, 20]),))
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertIn("dense_resource", asm)

    def testDynamicNNModule(self):
        mdl = SimpleParams()
        batch = torch.export.Dim("batch")
        exported = export(
            mdl, args=(torch.empty([128, 20]),), dynamic_shapes={"x": {0: batch}}
        )
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertIn(
            "func.func @main(%arg0: !torch.vtensor<[?,20],f32>) -> !torch.vtensor<[?,30],f32>",
            asm,
        )

    def testExternalParamsNNModule(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)
        exported = export(mdl, args=(torch.empty([128, 20]),))
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertNotIn("dense_resource", asm)
        self.assertIn("util.global.load", asm)

    def testTorchExportedProgram(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)
        prg = torch.export.export(mdl, args=(torch.empty([128, 20]),))
        exported = export(prg)
        exported.print_readable()
        asm = str(exported.mlir_module)
        self.assertNotIn("dense_resource", asm)
        self.assertIn(
            'util.global private @__auto.classifier.weight = #stream.parameter.named<"model"::"classifier.weight">',
            asm,
        )
        self.assertIn(
            'util.global private @__auto.classifier.bias = #stream.parameter.named<"model"::"classifier.bias">',
            asm,
        )

    def testCompiledModuleExportedProgram(self):
        class BasicModule(CompiledModule):
            ...

        exported = export(BasicModule)
        module_str = str(exported.mlir_module)
        print(module_str)
        self.assertIn("module @basic", module_str)

    def testUnsupportedExportedProgram(self):
        class UnsupportedExportType:
            ...

        with self.assertRaises(TypeError):
            export(UnsupportedExportType)


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
