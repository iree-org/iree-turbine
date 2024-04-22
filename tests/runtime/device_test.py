# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import threading

import torch

from iree.runtime import HalElementType

# Public API imports.
from shark_turbine.runtime import (
    Device,
)

# Internals.
from shark_turbine.runtime.device import (
    _CURRENT_THREAD,
    get_device_from_torch,
)

from shark_turbine.support.exceptions import *


class DeviceTest(unittest.TestCase):
    def test_create(self):
        d = Device("local-task")
        self.assertEqual(repr(d), "<Turbine Device: local-task>")

    def test_current_device(self):
        with self.assertRaises(NoCurrentDeviceError):
            Device.current()

        d1 = Device("local-task")
        d2 = Device("local-sync")
        with d1:
            self.assertIs(Device.current(), d1)

            with d2:
                self.assertIs(Device.current(), d2)

            self.assertIs(Device.current(), d1)

        with self.assertRaises(NoCurrentDeviceError):
            Device.current()

    def test_set_clear(self):
        d1 = Device("local-task")
        d2 = Device("local-sync")

        with self.assertRaises(MismatchedDeviceSetClearError):
            d1.clear()
        try:
            d1.set()
            self.assertIs(Device.current(), d1)
            with self.assertRaises(MismatchedDeviceSetClearError):
                d2.clear()
            d1.clear()
            with self.assertRaises(NoCurrentDeviceError):
                Device.current()
        finally:
            # Patch it back to the reset state for testing.
            _CURRENT_THREAD.stack = []

    def test_cached_devices_same_thread(self):
        d1 = Device("local-task")
        d2 = Device("local-task")
        self.assertIs(d1, d2)

    def test_cached_device_diff_thread(self):
        devices = [None, None]

        def run_t1():
            devices[0] = Device("local-task")

        def run_t2():
            devices[1] = Device("local-task")

        t1 = threading.Thread(target=run_t1)
        t2 = threading.Thread(target=run_t2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        self.assertIsNotNone(devices[0])
        self.assertIsNotNone(devices[1])
        self.assertIsNot(devices[0], devices[1])


# CPU is always available so we can enable this unconditionally.
class TorchCPUInterop(unittest.TestCase):
    def testFromTorchDevice(self):
        torch_device = torch.device("cpu")
        device1 = get_device_from_torch(torch_device)
        print(device1)
        self.assertIsNotNone(device1)
        device2 = get_device_from_torch(torch_device)
        self.assertIs(device1, device2)

    def testCpuDeviceCacheKey(self):
        d = get_device_from_torch(torch.device("cpu"))
        self.assertEqual(d.instance_cache_key, "local-task")
        self.assertEqual(d.type_cache_key, "local-task")

    def testImportExportTorchTensor(self):
        d = get_device_from_torch(torch.device("cpu"))
        cpu_tensor = torch.tensor([1, 2, 3], dtype=torch.int32, device="cpu")
        bv = d.import_torch_tensor(cpu_tensor)
        print(bv)
        self.assertEqual(bv.shape, [3])
        self.assertEqual(bv.element_type, HalElementType.SINT_32)
        meta_tensor = cpu_tensor.to(device="meta")
        readback_tensor = d.export_torch_tensor(bv, meta_tensor)
        torch.testing.assert_close(cpu_tensor, readback_tensor)

    def testCompilerFlags(self):
        d = get_device_from_torch(torch.device("cpu"))
        self.assertIn("--iree-hal-target-backends=llvm-cpu", d.compile_target_flags)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
