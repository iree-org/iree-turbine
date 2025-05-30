# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest

import torch

from iree.turbine.runtime.device import get_device_from_torch
from iree.turbine.ops.conv_fwd import conv_2d_nhwc_fhwc


class DeviceStreamTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            return
        self.stream0 = torch.cuda.current_stream(0)
        self.stream1 = torch.cuda.Stream(0)
        self.stream2 = torch.cuda.Stream(0)
        print(self.stream0.cuda_stream)
        print(self.stream0.stream_id)
        print(self.stream1.cuda_stream)
        print(self.stream1.stream_id)
        print(self.stream2.cuda_stream)
        print(self.stream2.stream_id)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires gpu.")
    def testDeviceCreation(self):
        d0 = get_device_from_torch(torch.device("cuda:0"))
        with torch.cuda.stream(self.stream1):
            d1 = get_device_from_torch(torch.device("cuda:0"))
        with torch.cuda.stream(self.stream2):
            d2 = get_device_from_torch(torch.device("cuda:0"))
        self.assertEqual(d0._s.torch_stream, self.stream0.cuda_stream)
        self.assertEqual(d1._s.torch_stream, self.stream1.cuda_stream)
        self.assertEqual(d2._s.torch_stream, self.stream2.cuda_stream)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires gpu.")
    def testEagerCustomOpMultiStream(self):
        """This test uses a CustomOp to validate multi-stream invocations and Device-Host copy.
        Heavier computations are used to ensure that GPU synchronization isn't trivialized by CPU overhead."""

        torch_device = torch.device("cuda:0")
        input0 = torch.ones(
            [16, 2048, 2048, 16], dtype=torch.float32, device=torch_device
        )
        input1 = torch.ones([4, 16, 16, 16], dtype=torch.float32, device=torch_device)
        input2 = torch.ones([3, 1, 1, 4], dtype=torch.float32, device=torch_device)
        self.stream1.wait_stream(torch.cuda.current_stream(0))
        # Do the computations once on the default stream to jit compile artifacts.
        res0 = conv_2d_nhwc_fhwc(input0, input1, [1, 1], [1, 1])
        _ = conv_2d_nhwc_fhwc(res0, input2, [1, 1], [1, 1])

        for _ in range(20):
            # Do first conv on stream 1
            with torch.cuda.stream(self.stream1):
                res0 = conv_2d_nhwc_fhwc(input0, input1, [1, 1], [1, 1])
                ev = torch.cuda.Event()
                ev.record(self.stream1)

            input0.record_stream(self.stream1)
            input1.record_stream(self.stream2)

            # Do second conv on stream 2, waiting for stream 1's corresponding conv
            with torch.cuda.stream(self.stream2):
                ev.wait()
                res = conv_2d_nhwc_fhwc(res0, input2, [1, 1], [1, 1])
            res0.record_stream(self.stream2)
            input2.record_stream(self.stream2)

        # move the last result from each stream to host without blocking
        with torch.cuda.stream(self.stream1):
            cpu_last_result_1 = res0.to(device="cpu", non_blocking=True)
        with torch.cuda.stream(self.stream2):
            cpu_last_result_2 = res.to(device="cpu", non_blocking=True)

        # validate the results match expectations
        self.assertEqual(cpu_last_result_1[0, 0, 0, 0].item(), 16**3)
        self.assertEqual(cpu_last_result_2[0, 0, 0, 0].item(), 4 * (16**3))


if __name__ == "__main__":
    unittest.main()
