# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.compiler.ir import (
    Context,
    Type as IrType,
)

import shark_turbine.dynamo.type_conversion as tc


class TypeConversionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.conv = tc.NativeTypeConverter(Context())

    def testPrimitives(self):
        self._compareNative("!torch.bool", "i1")
        self._compareNative("!torch.int", "i64")
        self._compareNative("!torch.float", "f64")

    def testSigned(self):
        self._compareNative("!torch.bool", "i1", signless=False)
        self._compareNative("!torch.int", "si64", signless=False)

    def testValueTensors(self):
        self._compareNative("!torch.vtensor<[2, 2],f32>", "tensor<2x2xf32>")
        self._compareNative("!torch.vtensor<[?, ?],f32>", "tensor<?x?xf32>")
        self._compareNative("!torch.vtensor<[],f32>", "tensor<f32>")

    def _compareNative(self, torch_str: str, native_str: str, *, signless: bool = True):
        with self.conv._context:
            torch_type = IrType.parse(torch_str)
        native_type = self.conv.torch_type_to_native(torch_type, signless=signless)
        self.assertEqual(str(native_type), native_str)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
