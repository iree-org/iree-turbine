# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

import iree.runtime as rt
import iree.compiler as ireec

from iree.turbine.aot import (
    export,
    externalize_module_parameters,
    save_module_parameters,
    ExternalTensorTrait,
    ParameterArchive,
    ParameterArchiveBuilder,
)


class SimpleParamsModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(20, 30)
        self.large_tensor = torch.rand([30, 50])
        self.dup_large_tensor = torch.rand([30, 50])

    def forward(self, x):
        result = self.classifier(x) + torch.tensor([1.0], dtype=torch.float32)
        result = torch.matmul(result, self.large_tensor + self.dup_large_tensor)
        return result


class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        return (input @ self.weight) + self.bias


class ExternalParamsTest(unittest.TestCase):
    def setUp(self):
        self.instance = rt.VmInstance()
        self.device = rt.get_device(ireec.core.DEFAULT_TESTING_DRIVER)
        self.config = rt.Config(device=self.device)

    def testSeparateWeightsAtRuntime(self):
        linear_module = LinearModule(4, 3).requires_grad_(False)
        externalize_module_parameters(linear_module)
        wt = linear_module.weight.data.contiguous()
        bias = linear_module.bias.data.contiguous()

        input = torch.randn(4)
        exported_module = export(linear_module, input)
        binary = exported_module.compile(save_to=None)

        idx = rt.ParameterIndex()
        idx.add_buffer("weight", wt.detach().numpy().tobytes())
        idx.add_buffer("bias", bias.detach().numpy().tobytes())

        config = rt.Config(driver_name="local-task")
        instance = config.vm_instance
        param_module = rt.create_io_parameters_module(
            instance,
            idx.create_provider(scope="model"),
        )

        vm_modules = rt.load_vm_modules(
            param_module,
            rt.create_hal_module(instance, config.device),
            rt.VmModule.copy_buffer(instance, binary.map_memory()),
            config=config,
        )

        m = vm_modules[-1]
        result_vm = m.main(input).to_host()
        result_torch = linear_module(input)
        torch.testing.assert_close(torch.from_numpy(result_vm), result_torch)


class ParamsTest(unittest.TestCase):
    def testCreateArchive(self):
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            m = SimpleParamsModule()
            save_module_parameters(file_path, m)
            # mmap=False is a bit nicer for tests on Windows because it doesn't
            # lock the file for an arbitrary duration.
            archive = ParameterArchive(file_path, mmap=False)
            items = dict(archive.items())
            weight = items["classifier.weight"].as_tensor()
            bias = items["classifier.bias"].as_tensor()
            torch.testing.assert_close(weight, m.classifier.weight)
            torch.testing.assert_close(bias, m.classifier.bias)

    def testRoundtripScalarTensor(self):
        # See: https://github.com/iree-org/iree-turbine/issues/29
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            orig_scalar = torch.tensor(0.5, dtype=torch.float32)
            builder = ParameterArchiveBuilder()
            builder.add_tensor("scalar", orig_scalar)
            builder.save(file_path)
            archive = ParameterArchive(file_path, mmap=False)
            items = dict(archive.items())
            scalar = items["scalar"].as_tensor()
            torch.testing.assert_close(orig_scalar, scalar)

    def testRoundtripScalarUint8(self):
        # See: https://github.com/iree-org/iree-turbine/issues/29
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            orig_scalar = torch.tensor(8, dtype=torch.uint8)
            builder = ParameterArchiveBuilder()
            builder.add_tensor("scalar", orig_scalar)
            builder.save(file_path)
            archive = ParameterArchive(file_path, mmap=False)
            items = dict(archive.items())
            scalar = items["scalar"].as_tensor()
            torch.testing.assert_close(orig_scalar, scalar)

    def testCreateArchiveWithPrefixScope(self):
        with tempfile.TemporaryDirectory() as td:
            file_path = Path(td) / "archive.irpa"
            m = SimpleParamsModule()
            save_module_parameters(file_path, m, prefix="foobar.model")
            # mmap=False is a bit nicer for tests on Windows because it doesn't
            # lock the file for an arbitrary duration.
            archive = ParameterArchive(file_path, mmap=False)
            items = dict(archive.items())
            weight = items["foobar.model.classifier.weight"].as_tensor()
            bias = items["foobar.model.classifier.bias"].as_tensor()
            torch.testing.assert_close(weight, m.classifier.weight)
            torch.testing.assert_close(bias, m.classifier.bias)

    def testExportExternalized(self):
        m = SimpleParamsModule()
        externalize_module_parameters(m)
        output = export(m, args=(torch.empty([128, 20]),))
        asm = str(output.mlir_module)
        self.assertIn(
            'util.global private @__auto.classifier.weight = #stream.parameter.named<"model"::"classifier.weight">',
            asm,
        )
        self.assertIn(
            'util.global private @__auto.classifier.bias = #stream.parameter.named<"model"::"classifier.bias">',
            asm,
        )
        # Verify that the small tensor is inlined.
        self.assertIn("torch.vtensor.literal(dense<1.000000e+00> : tensor<1xf32>)", asm)
        # Verify that the large tensors are named uniquely and lifted.
        self.assertIn("@__auto.constant_30_50_torch.float32 =", asm)
        self.assertIn("@__auto.constant_30_50_torch.float32$1 =", asm)


class ExternalTensorTest(unittest.TestCase):
    def testExternalTensorTrait(self):
        t = torch.ones([2, 3], dtype=torch.float32)
        trait = ExternalTensorTrait(external_name="foobar", external_scope="test")
        self.assertIsNone(trait.get(t))
        trait.set(t)
        self.assertIs(ExternalTensorTrait.get(t), trait)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
