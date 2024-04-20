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

from shark_turbine.aot import (
    export,
    externalize_module_parameters,
    save_module_parameters,
    ExternalTensorTrait,
    ParameterArchive,
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


class ParamsTest(unittest.TestCase):
    def testCreateArchive(self):
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".irpa") as f:
            file_path = Path(f.name)
            try:
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
            finally:
                file_path.unlink()

    def testCreateArchiveWithPrefixScope(self):
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".irpa") as f:
            file_path = Path(f.name)
            try:
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
            finally:
                file_path.unlink()

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
