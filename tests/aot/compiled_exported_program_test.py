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

from iree.turbine.aot import *
from iree.turbine.aot.builtins import *


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

    def testParametersAsExplicitGlobals(self):
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
        self.assertIn("util.global private @_params.classifier.weight", module_str)
        self.assertIn("util.global private @_params.classifier.bias", module_str)
        # Should only be two.
        self.assertEqual(2, module_str.count("util.global private"))
        # And two loads each loads.
        self.assertEqual(
            2, module_str.count("util.global.load @_params.classifier.weight")
        )
        self.assertEqual(
            2, module_str.count("util.global.load @_params.classifier.bias")
        )

    def testParametersAsGlobalsViaExternalizeModuleParameters(self):
        mdl = SimpleParams()
        externalize_module_parameters(mdl)

        fxb = FxProgramsBuilder(mdl)

        @fxb.export_program(
            args=(torch.empty([128, 20]),),
        )
        def _compute1(module, x):
            return module.forward(x)

        class ParamsAsGlobalsModule(CompiledModule):
            compute1 = _compute1
            compute2 = _compute1

        inst = ParamsAsGlobalsModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("util.global private @__auto.classifier.weight", module_str)
        self.assertIn("util.global private @__auto.classifier.bias", module_str)

        # It's clunky to verify ordering, but we explicitly guarantee that
        # implicitly exported globals are emitted in order of declaration,
        # preceeding all functions.
        g1_index = module_str.index("util.global private @__auto.classifier.weight")
        g2_index = module_str.index("util.global private @__auto.classifier.bias")
        f_index = module_str.index("func")
        self.assertGreater(g2_index, g1_index)
        self.assertGreater(f_index, g2_index)

        # Should only be two.
        self.assertEqual(2, module_str.count("util.global private"))
        # And two loads each loads.
        self.assertEqual(
            2, module_str.count("util.global.load @__auto.classifier.weight")
        )
        self.assertEqual(
            2, module_str.count("util.global.load @__auto.classifier.bias")
        )

    def testBuffersAsGlobals(self):
        fxb = FxProgramsBuilder(SimpleBuffers())

        @fxb.export_program(args=(torch.empty([128]),))
        def _compute1(module, x):
            return module.forward(x)

        class BuffersAsGlobalsModule(CompiledModule):
            buffers = export_buffers(fxb.root_module, mutable=True)
            compute1 = _compute1

        inst = BuffersAsGlobalsModule(context=Context(), import_to="import")
        module_str = str(CompiledModule.get_mlir_module(inst))
        self.assertIn("util.global private mutable @_buffers.buf", module_str)
        self.assertIn("%_buffers.buf = util.global.load @_buffers.buf", module_str)
        self.assertIn("util.global.store", module_str)


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


class SimpleBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.randn(1))

    def forward(self, x: torch.Tensor):
        sumx = (x).sum()
        output = x * self.buf
        self.buf.copy_(sumx)
        return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
