# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
import unittest
import logging
from io import StringIO

from pathlib import Path
from parameterized import parameterized_class

import torch

from iree.turbine.aot.params import (
    ParameterArchiveBuilder,
)

from iree.turbine.runtime import (
    Launchable,
)

from iree.turbine.support.logging import runtime_logger as logger

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
util.global private @param = #flow.parameter.named<"param"> : tensor<4xi32>
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
    num_gpus = torch.cuda.device_count()
    devices.append([torch.device("cuda:0")])
    if num_gpus > 1:
        devices.append([torch.device("cuda:1")])


@parameterized_class(["device"], devices)
class LaunchableTest(unittest.TestCase):
    def testLaunchJit(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM, entry_point="main")
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)

    def testLaunchPreload(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM, entry_point="main")
        launch.preload(self.device)
        launch._loader = None  # Don't let it load anything more.
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)

    def testLaunchParamsWithoutParams(self):
        launch = Launchable.jit_compile(MLIR_PARAMS_ASM, entry_point="main")
        with self.assertRaisesRegex(
            ValueError, "required module 'io_parameters' not registered"
        ):
            launch.preload(self.device)

    def testLaunchParams(self):
        param_archive = ParameterArchiveBuilder()
        param_archive.add_tensor("param", torch.tensor([2, 4, 6, 8], dtype=torch.int32))
        provider = param_archive.index.create_provider()

        launch = Launchable.jit_compile(
            MLIR_PARAMS_ASM, parameter_providers=[provider], entry_point="main"
        )
        launch.preload(self.device)
        t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(self.device)
        t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(self.device)
        result = launch(t1, t2)
        expected = torch.tensor([12, 44, 96, 168], dtype=torch.int32).to(self.device)
        torch.testing.assert_close(expected, result)

    def testLaunchWithFileCache(self):
        """Tests that vmfb files are getting stored to and loaded from a file_cache."""
        with tempfile.TemporaryDirectory() as td:
            temp_cache = Path(td)
            launch_0 = Launchable.jit_compile(
                MLIR_NO_PARAMS_ASM, file_cache_dir=temp_cache, entry_point="main"
            )
            launch_0.preload(self.device)
            files = [f for f in temp_cache.glob("*.vmfb")]
            self.assertTrue(len(files) == 1, "Expected a single cached vmfb.")
            # Make a new launchable from the same source and verify it loads from the cache.
            launch_1 = Launchable.jit_compile(
                MLIR_NO_PARAMS_ASM, file_cache_dir=temp_cache, entry_point="main"
            )
            # This is a bit hacky: Intercept the logs to check for whether the vmfb gets loaded.
            old_level = logger.level
            logger.setLevel(logging.DEBUG)
            intercept_stream = StringIO()
            intercept_handler = logging.StreamHandler(intercept_stream)
            logger.addHandler(intercept_handler)
            launch_1.preload(self.device)
            self.assertIn("Loading vmfb from cache:", intercept_stream.getvalue())
            logger.removeHandler(intercept_handler)
            intercept_stream.close()
            logger.setLevel(old_level)

    def testLaunchFileCacheOnly(self):
        with tempfile.TemporaryDirectory() as td:
            temp_cache = Path(td)
            launch_no_jit = Launchable.from_file_cache_only(file_cache_dir=temp_cache)
            # check that the empty cache causes failure on preload
            with self.assertRaises(RuntimeError):
                launch_no_jit.preload(self.device)
            # get a launchable with file cache and preload
            launch_jit = Launchable.jit_compile(
                MLIR_NO_PARAMS_ASM, file_cache_dir=temp_cache, entry_point="main"
            )
            launch_jit.preload(self.device)
            # preload using the no_jit launchable
            launch_no_jit.preload(self.device)


@unittest.skipIf(
    len(devices) == 1,
    "Tests are redundant if not running on multiple devices.",
)
class SameLaunchableDifferentDevicesTest(unittest.TestCase):
    """
    These are roughly equivalent to the previous tests, but the Launchable persists between launches to test the caching.
    """

    def testLaunchJit(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM, entry_point="main")
        for d in devices:
            device = d[0]
            t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(device)
            t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(device)
            result = launch(t1, t2)
            expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(device)
            torch.testing.assert_close(expected, result)
        self.assertEqual(len(devices), len(launch._target_binaries.keys()))

    def testLaunchPreload(self):
        launch = Launchable.jit_compile(MLIR_NO_PARAMS_ASM, entry_point="main")
        for d in devices:
            device = d[0]
            launch.preload(device)
        launch._loader = None  # Don't let it load anything more.
        for d in devices:
            device = d[0]
            t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(device)
            t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(device)
            result = launch(t1, t2)
            expected = torch.tensor([10, 40, 90, 160], dtype=torch.int32).to(device)
            torch.testing.assert_close(expected, result)
        self.assertEqual(len(devices), len(launch._target_binaries.keys()))

    def testLaunchParams(self):
        param_archive = ParameterArchiveBuilder()
        param_archive.add_tensor("param", torch.tensor([2, 4, 6, 8], dtype=torch.int32))
        provider = param_archive.index.create_provider()

        launch = Launchable.jit_compile(
            MLIR_PARAMS_ASM, parameter_providers=[provider], entry_point="main"
        )
        for d in devices:
            device = d[0]
            launch.preload(device)
            t1 = torch.tensor([1, 2, 3, 4], dtype=torch.int32).to(device)
            t2 = torch.tensor([10, 20, 30, 40], dtype=torch.int32).to(device)
            result = launch(t1, t2)
            expected = torch.tensor([12, 44, 96, 168], dtype=torch.int32).to(device)
            torch.testing.assert_close(expected, result)
        self.assertEqual(len(devices), len(launch._target_binaries.keys()))


if __name__ == "__main__":
    unittest.main()
