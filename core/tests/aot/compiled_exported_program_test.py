# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch.nn as nn

from iree.compiler.ir import (
    Context,
)

from shark_turbine.aot import *
from shark_turbine.aot.builtins import *


class TorchExportTests(unittest.TestCase):
    def testImportPhases(self):
        class MyModule(torch.nn.Module):
            def forward(self):
                ...

        fxb = FxProgramsBuilder(MyModule())

        @fxb.export_program(
            args=([torch.empty([3, 2]), torch.empty([1, 2])],),
            kwargs={"foobar": torch.empty([3, 1])},
        )
        def compute(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        class ExportedProcModule(CompiledModule):
            _compute = compute

            def foobar(
                self,
                t1=AbstractTensor(3, 2),
                t2=AbstractTensor(1, 2),
                t3=AbstractTensor(3, 1),
            ):
                return self._compute(t1, t2, foobar=t3)

        inst = ExportedProcModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("func.func private @_compute", module_str)
        self.assertIn("func.func @foobar", module_str)

    def testMultiPublic(self):
        class MyModule(torch.nn.Module):
            def forward(self):
                ...

        fxb = FxProgramsBuilder(MyModule())

        @fxb.export_program(
            args=([torch.empty([3, 2]), torch.empty([1, 2])],),
            kwargs={"foobar": torch.empty([3, 1])},
        )
        def _compute1(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        @fxb.export_program(
            args=([torch.empty([5]), torch.empty([5])],),
            kwargs={"foobar": torch.empty([5])},
        )
        def _compute2(module, inputs, *, foobar):
            t1 = inputs[0]
            t2 = inputs[1]
            t3 = t1 + t2 + foobar
            return [t3 * t3, foobar]

        class ExportedPublicModule(CompiledModule):
            compute1 = _compute1
            compute2 = _compute2

        inst = ExportedPublicModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("func.func @compute1", module_str)
        self.assertIn("func.func @compute2", module_str)

    def testParametersAsGlobals(self):
        fxb = FxProgramsBuilder(SimpleParams())

        @fxb.export_program(
            args=(torch.empty([128, 20]),),
        )
        def _compute1(module, x):
            return module.forward(x)

        class ParamsAsGlobalsModule(CompiledModule):
            params = export_parameters(fxb.root_module)
            compute1 = _compute1
            compute2 = _compute1

        inst = ParamsAsGlobalsModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "util.global private @_params.classifier.weight {noinline}", module_str
        )
        self.assertIn(
            "util.global private @_params.classifier.bias {noinline}", module_str
        )
        # Should only be two.
        self.assertEqual(2, module_str.count("util.global private"))
        # And two loads each loads.
        self.assertEqual(
            2, module_str.count("util.global.load @_params.classifier.weight")
        )
        self.assertEqual(
            2, module_str.count("util.global.load @_params.classifier.bias")
        )


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
