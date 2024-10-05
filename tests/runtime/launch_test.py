# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from parameterized import parameterized_class
import torch
import unittest

from iree.turbine.aot.params import (
    ParameterArchiveBuilder,
)

from iree.turbine.runtime import (
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

# TODO: Move this to a common utility controlled by project wide env vars.
devices = [[torch.device("cpu")]]
if torch.cuda.is_available():
    devices.append([torch.device("cuda:0")])


@parameterized_class(["device"], devices)
class LaunchableTest(unittest.TestCase):
    def testLaunchJit(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM)
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)

    def testLaunchPreload(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM)
        launch.preload(self.device)
        launch._loader = None  # Don't let it load anything more.
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)

    def testLaunchParamsWithoutParams(self):
        launch = Launchable.jit_compile(MLIR_PARAMS_ASM)
        with self.assertRaisesRegex(
            ValueError, "required module 'io_parameters' not registered"
        ):
            launch.preload(self.device)

    def testLaunchParams(self):
        param_archive = ParameterArchiveBuilder()
        param_archive.add_tensor("param", torch.tensor([2, 4, 6, 8], dtype=torch.int32))
        provider = param_archive.index.create_provider()

        launch = Launchable.jit_compile(MLIR_PARAMS_ASM, parameter_providers=[provider])
        launch.preload(self.device)
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([12, 44, 96, 168], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)


if __name__ == "__main__":
    unittest.main()
