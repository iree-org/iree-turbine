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


class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)

    def forward(self, x):
        return self.classifier(x)


class GlobalsTest(unittest.TestCase):
    def testGlobalParameters(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule):
            params = export_parameters(m)
            compute = jittable(m.forward)

            def run(self, x=AbstractTensor(128, 20)):
                return self.compute(x)

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("util.global private @_params.classifier.weight", module_str)
        self.assertIn("util.global private @_params.classifier.bias", module_str)

    def testGlobalLoadFromPyTree(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule):
            params = export_parameters(m)

            def read_params(self):
                return self.params

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "%_params.classifier.weight = util.global.load @_params.classifier.weight",
            module_str,
        )
        self.assertIn(
            "%_params.classifier.bias = util.global.load @_params.classifier.bias",
            module_str,
        )

    def testGlobalLoadFromPyLeaf(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule):
            params = export_parameters(m)

            def read_weight(self):
                return self.params["classifier.weight"]

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "%_params.classifier.weight = util.global.load @_params.classifier.weight",
            module_str,
        )

    def testGlobalStoreFromPyTree(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule):
            params = export_parameters(m, mutable=True)

            def update_params(me, updates=abstractify(params)):
                self.assertIn("classifier.weight", updates)
                self.assertIn("classifier.bias", updates)
                me.params = updates

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertRegex(
            module_str, "util.global.store %.*, @_params.classifier.weight"
        )
        self.assertRegex(module_str, "util.global.store %.*, @_params.classifier.bias")

    def testGlobalStoreFromLeaf(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule):
            params = export_parameters(m, mutable=True)

            def update_bias(self, new_bias=abstractify(params["classifier.bias"])):
                self.params["classifier.bias"] = new_bias

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertRegex(module_str, "util.global.store %.*, @_params.classifier.bias")

    def testExportSingleGlobalTensor(self):
        state_example = torch.randn(3, 11)

        class SingleState(CompiledModule):
            state0 = export_global(state_example, name="global")

            def read_state(self):
                return self.state0

        inst = SingleState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("util.global private @_state0.global", module_str)
        self.assertIn("%_state0.global = util.global.load @_state0.global", module_str)

    def testExportTreeGlobalTensors(self):
        state_example = {
            "data": torch.randn(3, 11),
            "seq": [
                torch.randn(1),
                torch.randn(2),
                torch.randn(3),
            ],
        }

        class SingleState(CompiledModule):
            state0 = export_global_tree(state_example)

            def read_state(self):
                return self.state0

        inst = SingleState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("util.global private @_state0.seq.0", module_str)
        self.assertIn("util.global private @_state0.seq.1", module_str)
        self.assertIn("util.global private @_state0.seq.2", module_str)
        self.assertIn("util.global private @_state0.data", module_str)
        self.assertIn("%_state0.data = util.global.load @_state0.data", module_str)
        self.assertIn("%_state0.seq.0 = util.global.load @_state0.seq.0", module_str)
        self.assertIn("%_state0.seq.1 = util.global.load @_state0.seq.1", module_str)
        self.assertIn("%_state0.seq.2 = util.global.load @_state0.seq.2", module_str)

    def testExportGlobalScalars(self):
        class ScalarState(CompiledModule):
            state_index = export_global(AbstractIndex, mutable=True)
            state_f32 = export_global(AbstractF32, mutable=True)
            state_f64 = export_global(AbstractF64, mutable=True)
            state_i32 = export_global(AbstractI32, mutable=True)
            state_i64 = export_global(AbstractI64, mutable=True)
            state_bool = export_global(AbstractBool, mutable=True)

            def read(self):
                return (
                    self.state_index,
                    self.state_f32,
                    self.state_f64,
                    self.state_i32,
                    self.state_i64,
                    self.state_bool,
                )

        inst = ScalarState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("@_state_index.global {noinline} = 0 : index", module_str)
        self.assertIn("@_state_f32.global {noinline} = 0.000000e+00 : f32", module_str)
        self.assertIn("@_state_f64.global {noinline} = 0.000000e+00 : f64", module_str)
        self.assertIn("@_state_i32.global {noinline} = 0 : i32", module_str)
        self.assertIn("@_state_i64.global {noinline} = 0 : i64", module_str)
        self.assertIn("@_state_bool.global {noinline} = false", module_str)

    def testInheritExportScalars(self):
        class BaseState(CompiledModule):
            state_index = export_global(AbstractIndex, mutable=True)
            state_f32 = export_global(AbstractF32, mutable=True)

            def read(self):
                return (self.state_index, self.state_f32)

        class DerivedState(BaseState):
            pass

        inst = DerivedState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("@_state_index.global {noinline} = 0 : index", module_str)
        self.assertIn("@_state_f32.global {noinline} = 0.000000e+00 : f32", module_str)

    def testInheritOverrideBase(self):
        class BaseState(CompiledModule):
            state_index = export_global(AbstractIndex, mutable=True)
            state_f32 = export_global(AbstractF32, mutable=True)

            def read(self):
                return (self.state_index, self.state_f32)

        class DerivedState(BaseState):
            def read(self):
                return self.state_index

        inst = DerivedState(context=Context(), import_to="full")
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("@_state_index.global {noinline} = 0 : index", module_str)
        self.assertNotIn(
            "@_state_f32.global {noinline} = 0.000000e+00 : f32", module_str
        )
        self.assertIn("return %_state_index.global : index", module_str)

    def testInheritExportModules(self):
        m = SimpleParams()

        class BaseModule(CompiledModule):
            params = export_parameters(m, mutable=True)

            def update_params(me, updates=abstractify(params)):
                self.assertIn("classifier.weight", updates)
                self.assertIn("classifier.bias", updates)
                me.params = updates

        class DerivedModule(BaseModule):
            pass

        inst = DerivedModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertRegex(
            module_str, "util.global.store %.*, @_params.classifier.weight"
        )
        self.assertRegex(module_str, "util.global.store %.*, @_params.classifier.bias")

    def testUpdateGlobalStateTree(self):
        state_example = {
            "data": torch.randn(3, 11),
            "seq": [
                torch.randn(1).to(torch.int32),
                torch.randn(2).to(torch.float64),
                torch.randn(3).to(torch.int64),
            ],
        }

        class SingleState(CompiledModule):
            state0 = export_global_tree(abstractify(state_example), mutable=True)

            def read_state(self, updates=abstractify(state_example)):
                self.state0 = updates

        inst = SingleState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "util.global private mutable @_state0.seq.0 {noinline} = dense<0> : tensor<1xi32>",
            module_str,
        )
        self.assertIn(
            "util.global private mutable @_state0.seq.1 {noinline} = dense<0.000000e+00> : tensor<2xf64>",
            module_str,
        )
        self.assertIn(
            "util.global private mutable @_state0.seq.2 {noinline} = dense<0> : tensor<3xi64>",
            module_str,
        )
        self.assertIn("util.global private mutable @_state0.data", module_str)
        self.assertRegex(module_str, "util.global.store %.*, @_state0.data")
        self.assertRegex(module_str, "util.global.store %.*, @_state0.seq.0")
        self.assertRegex(module_str, "util.global.store %.*, @_state0.seq.1")
        self.assertRegex(module_str, "util.global.store %.*, @_state0.seq.2")

    def testTensorUpdateGlobal(self):
        state_example = torch.randn(5, 20)
        update_example = torch.randn(1, 20)

        class UpdateState(CompiledModule):
            state0 = export_global(state_example, mutable=True)

            def tensor_update_state(self, update=abstractify(update_example)):
                return IREE.tensor_update(self.state0, update, 0, 0)

        inst = UpdateState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertRegex(
            module_str,
            "flow.tensor.update %.*, %_state0.global\\[%c0, %c0\\] : tensor<1x20xf32> -> %_state0.global as tensor<5x20xf32>",
        )

    def testTensorUpdateGlobalReturnNone(self):
        state_example = torch.randn(5, 20, 4)
        update_example = torch.randn(1, 1, 4)

        class UpdateState(CompiledModule):
            state0 = export_global(state_example, mutable=True)

            def tensor_update_state(self, update=abstractify(update_example)):
                thing = []
                self.state0 = IREE.tensor_update(self.state0, update, 4, 0, 0)
                return None

        inst = UpdateState(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn("flow.tensor.update", module_str)

    def testExternalGlobalParametersDefaults(self):
        m = SimpleParams()

        class GlobalModule(
            CompiledModule, export_name="external_global_parameters_defaults"
        ):
            params = export_parameters(m, external=True)
            compute = jittable(m.forward)

            def run(self, x=AbstractTensor(128, 20)):
                return self.compute(x)

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            '#stream.parameter.named<"model"::"params.classifier.weight"> : tensor<30x20xf32>',
            module_str,
        )
        self.assertIn(
            '#stream.parameter.named<"model"::"params.classifier.bias"> : tensor<30xf32>',
            module_str,
        )

    def testExternalGlobalParametersExplicit(self):
        m = SimpleParams()

        class GlobalModule(
            CompiledModule, export_name="external_global_parameters_explicit"
        ):
            params = export_parameters(
                m, external=True, external_scope="foo", name_mapper=lambda s: s.upper()
            )
            compute = jittable(m.forward)

            def run(self, x=AbstractTensor(128, 20)):
                return self.compute(x)

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            '#stream.parameter.named<"foo"::"PARAMS.CLASSIFIER.WEIGHT"> : tensor<30x20xf32>',
            module_str,
        )
        self.assertIn(
            '#stream.parameter.named<"foo"::"PARAMS.CLASSIFIER.BIAS"> : tensor<30xf32>',
            module_str,
        )

    def testExternalGlobalParametersMapDict(self):
        m = SimpleParams()
        mapper = {
            "params.classifier.weight": "WEIGHT",
        }

        class GlobalModule(
            CompiledModule, export_name="external_global_parameters_map_dict"
        ):
            params = export_parameters(
                m, external=True, external_scope="foo", name_mapper=mapper.get
            )
            compute = jittable(m.forward)

            def run(self, x=AbstractTensor(128, 20)):
                return self.compute(x)

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            '#stream.parameter.named<"foo"::"WEIGHT"> : tensor<30x20xf32>',
            module_str,
        )
        self.assertIn(
            '#stream.parameter.named<"foo"::"params.classifier.bias"> : tensor<30xf32>',
            module_str,
        )

    def testUninitializedParameters(self):
        m = SimpleParams()

        class GlobalModule(CompiledModule, export_name="uninitialized_parameters"):
            params = export_parameters(m, uninitialized=True, mutable=True)
            y = export_global(AbstractF32, uninitialized=True, mutable=True)
            compute = jittable(m.forward)

            def run(self, x=AbstractTensor(128, 20)):
                return self.compute(x), self.y

        inst = GlobalModule(context=Context())
        module_str = str(CompiledModule.get_mlir_module(inst))
        print(module_str)
        self.assertIn(
            "#util.uninitialized : tensor<30x20xf32>",
            module_str,
        )
        self.assertIn(
            "#util.uninitialized : f32",
            module_str,
        )

    def testUnsupportedCombinations(self):
        with self.assertRaisesRegex(ValueError, "mutable=True"):
            export_global(AbstractF32, uninitialized=True)
        with self.assertRaisesRegex(ValueError, "external=True"):
            export_global(AbstractF32, external=True, uninitialized=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
