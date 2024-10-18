# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import unittest

import torch

from iree.turbine.aot import (
    DeviceAffinity,
    export,
    FxProgramsBuilder,
)


class FxProgramsTestDevice(unittest.TestCase):
    def test_argument_device_affinities(self):
        class Module(torch.nn.Module):
            def main(self, x1, x2):
                return x1, x2

        args = (
            torch.empty(2, 3, dtype=torch.int8),
            torch.empty(4, 5, dtype=torch.int8),
        )
        fxb = FxProgramsBuilder(Module())

        @fxb.export_program(
            args=args,
            arg_device={0: DeviceAffinity(0), 1: DeviceAffinity(1)},
        )
        def main(module: Module, x1, x2):
            return module.main(x1, x2)

        output = export(fxb)
        asm = str(output.mlir_module)
        self.assertRegex(
            asm,
            (
                "func.func @main\("
                "%.+: !torch.vtensor<\[2,3\],si8> {iree.abi.affinity = #hal.device.promise<@__device_0>}, "
                "%.+: !torch.vtensor<\[4,5\],si8> {iree.abi.affinity = #hal.device.promise<@__device_1>}\)"
            ),
        )
