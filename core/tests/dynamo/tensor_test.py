# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import time
import unittest

import numpy as np
import torch

# Public API imports.
from shark_turbine.runtime import Device
from shark_turbine.dynamo import TurbineMode, DeviceTensor


class TensorTest(unittest.TestCase):
    def setUp(self):
        self.mode = TurbineMode()
        self.mode.__enter__()
        Device("local-task").set()

    def tearDown(self) -> None:
        Device.current().clear()
        self.mode.__exit__(None, None, None)

    @unittest.expectedFailure
    def test_explicit_construct(self):
        size = (2, 2)
        t1 = DeviceTensor(size, torch.float32, np.ones(size))
        t2 = DeviceTensor(
            size, torch.float32, np.arange(size[0] * size[1]).reshape(size)
        )
        print("Inputs:")
        print(t1)
        print(t2)

    def test_async_copy_from_host(self):
        t1 = torch.empty(4, device="turbine")
        ar = np.arange(4, dtype=np.float32)
        t1._async_copy_from_host(ar)
        np.testing.assert_array_equal(t1.cpu(), ar)

    def test_cpu_to(self):
        t_cpu = torch.arange(4).cpu()
        t_t = t_cpu.to(device="turbine")
        np.testing.assert_array_equal(t_cpu, t_t.cpu())

    def test_factory_function_empty(self):
        # Factory functions
        t1 = torch.empty(4, device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)

    def test_factory_function_empty_tuple_size(self):
        # TODO: Test some invariants vs just printing.
        t1 = torch.empty((4, 4), device="turbine")
        print("Empty Tensor (un-initialized memory!):")
        print(t1)
        print(t1.buffer_view)
        print(t1.to("cpu"))
        print(t1.cpu())

    def test_factory_function_zeros(self):
        t1 = torch.zeros(2, 3, device="turbine")
        np.testing.assert_array_equal(t1.cpu(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_factory_function_ones(self):
        t1 = torch.ones(2, 3, device="turbine")
        np.testing.assert_array_equal(t1.cpu(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_factory_arange(self):
        t1 = torch.arange(4, device="turbine", dtype=torch.float32)
        ar = np.arange(4, dtype=np.float32)
        np.testing.assert_array_equal(t1.cpu(), ar)

    def test_factory_rand(self):
        t1 = torch.rand(4, device="turbine", dtype=torch.float32)
        print(t1.cpu())

    def test_binary_op(self):
        t1 = 5.3 * torch.ones(2, 3).to(device="turbine")
        t2 = 2.3 * torch.ones(2, 3).to(device="turbine")
        t3 = t1 * t2
        np.testing.assert_allclose(
            t3.cpu(), [[12.19, 12.19, 12.19], [12.19, 12.19, 12.19]]
        )

    def test_unary_op(self):
        t1 = -5.3 * torch.ones(2, 3).to(device="turbine")
        t2 = torch.abs(t1)
        np.testing.assert_allclose(t2.cpu(), [[5.3, 5.3, 5.3], [5.3, 5.3, 5.3]])

    def test_nn_linear(self):
        m = torch.nn.Linear(20, 30)
        input = torch.randn(128, 20)
        ref_output = m(input)
        m.to("turbine")
        input = input.to("turbine")
        turbine_output = m(input)
        np.testing.assert_allclose(
            turbine_output.cpu(), ref_output.detach().numpy(), atol=1e-6
        )

    def test_nn_MLP(self):
        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer0 = torch.nn.Linear(64, 32, bias=True)
                self.layer1 = torch.nn.Linear(32, 16, bias=True)
                self.layer2 = torch.nn.Linear(16, 7, bias=True)
                self.layer3 = torch.nn.Linear(7, 7, bias=True)

            def forward(self, x: torch.Tensor):
                x = self.layer0(x)
                x = torch.sigmoid(x)
                x = self.layer1(x)
                x = torch.sigmoid(x)
                x = self.layer2(x)
                x = torch.sigmoid(x)
                x = self.layer3(x)
                return x

        m = MLP()
        input = torch.randn(16, 64)
        ref_output = m(input)
        m.to("turbine")
        input = input.to("turbine")
        turbine_output = m(input)
        np.testing.assert_allclose(
            turbine_output.cpu(), ref_output.detach().numpy(), atol=1e-6
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
