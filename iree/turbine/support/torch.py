# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


def has_torch_device(d: str) -> bool:
    device = torch.device(d)
    if device.type == "cpu":
        return True
    if (
        device.type == "cuda"
        and torch.cuda.is_available()
        and (device.index is None or device.index < torch.cuda.device_count())
    ):
        return True

    return False


def torch_device_equal(d1: torch.device, d2: torch.device) -> bool:
    if d1.type == "cuda":
        # Somehow torch considers "cuda" and "cuda:0" different devices.
        # Even when the default device is "cuda:0".
        default_device = torch.get_default_device()
        default_cuda_index = 0
        if default_device.type == "cuda" and default_device.index is not None:
            default_cuda_index = default_device.index
        i1 = d1.index if d1.index is not None else default_cuda_index
        i2 = d2.index if d2.index is not None else default_cuda_index
        return d1.type == d2.type and i1 == i2
    return d1 == d2
