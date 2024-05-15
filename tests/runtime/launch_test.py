# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import unittest

from shark_turbine.aot.params import (
    ParameterArchiveBuilder,
)

from shark_turbine.runtime import (
    Launchable,
)

MLIR_NO_PARAMS_ASM = r"""
module @test_module {
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %0 = arith.muli %arg0, %arg1 : tensor<4xi32>
    return %0 : tensor<4xi32>
}
}
"""

MLIR_PARAMS_ASM = r"""
module @test_module {
util.global private @param = #stream.parameter.named<"param"> : tensor<4xi32>
func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %0 = arith.muli %arg0, %arg1 : tensor<4xi32>
    %param = util.global.load @param : tensor<4xi32>
    %1 = arith.addi %0, %param : tensor<4xi32>
    return %1 : tensor<4xi32>
}
}
"""


class LaunchableTest(unittest.TestCase):
    def testLaunchJit(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM)
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32)
        torch.testing.assert_close(expected, result)

    def testLaunchPreload(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM)
        launch.preload(torch.device("cpu"))
        launch._loader = None  # Don't let it load anything more.
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32)
        torch.testing.assert_close(expected, result)

    def testLaunchParamsWithoutParams(self):
        launch = Launchable.jit_compile(MLIR_PARAMS_ASM)
        with self.assertRaisesRegex(
            ValueError, "required module 'io_parameters' not registered"
        ):
            launch.preload(torch.device("cpu"))

    def testLaunchParams(self):
        param_archive = ParameterArchiveBuilder()
        param_archive.add_tensor("param", torch.tensor([2, 4, 6, 8], dtype=torch.int32))
        provider = param_archive.index.create_provider()

        launch = Launchable.jit_compile(MLIR_PARAMS_ASM, parameter_providers=[provider])
        launch.preload(torch.device("cpu"))
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        result = launch(t1, t2)
        expected = torch.tensor([12, 44, 96, 168], dtype=torch.int32)
        torch.testing.assert_close(expected, result)


if __name__ == "__main__":
    unittest.main()
