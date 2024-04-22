# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch

from shark_turbine.runtime.op_reg import *

from shark_turbine.runtime.op_reg.compiler import _testing_get_cache_size


class KernelRegTest(unittest.TestCase):
    def testRegistrationDispatchAndCache(self):
        @CustomOp.register
        class identity(CustomOp):
            signature = "test_identity(Tensor self) -> Tensor"

            def select(self, ksel: KernelSelection):
                x = ksel.arg_tensor(0)
                ksel.return_tensor(x.t)

            def generate(self, ksel: KernelSelection, kb: KernelBuilder):
                # This just yields the IR value of kernel input as the output.
                # Effectively in eager mode, this is a `return` from the kernel
                # function.
                kb.yield_results(kb.arg_bindings[0])

        self.assertIsNotNone(torch.ops.turbine.test_identity)

        start_compile_count = _testing_get_cache_size()

        # Make sure that the meta registration works.
        t = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="meta")
        result = identity(t)
        self.assertListEqual(list(result.shape), [1, 3])
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(t.device.type, "meta")
        # Meta dispatch should not trigger compilation.
        self.assertEqual(_testing_get_cache_size(), start_compile_count)

        # Make sure that CPU dispatch works.
        t = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        result = identity(t)
        print("CPU result:", result)
        torch.testing.assert_close(result, t)
        # Novel execution should compile a new kernel.
        self.assertEqual(_testing_get_cache_size(), start_compile_count + 1)

        # Second run of the same kernel should serve out of cache.
        result = identity(t)
        torch.testing.assert_close(result, t)
        # Repeated execution should use a cached kernel.
        self.assertEqual(_testing_get_cache_size(), start_compile_count + 1)

        # It should recompile for different dtype.
        t = torch.tensor([[1, 2, 3]], dtype=torch.int16)
        result = identity(t)
        print("CPU result:", result)
        torch.testing.assert_close(result, t)
        # Novel execution should compile a new kernel.
        self.assertEqual(_testing_get_cache_size(), start_compile_count + 2)

        # It should recompile for different rank.
        t = torch.tensor([1, 2, 3], dtype=torch.int16)
        result = identity(t)
        print("CPU result:", result)
        torch.testing.assert_close(result, t)
        # Novel execution should compile a new kernel.
        self.assertEqual(_testing_get_cache_size(), start_compile_count + 3)

        # It should serve out of cache for same-rank but different dims.
        t = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int16)
        result = identity(t)
        print("CPU result:", result)
        torch.testing.assert_close(result, t)
        self.assertEqual(_testing_get_cache_size(), start_compile_count + 3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
