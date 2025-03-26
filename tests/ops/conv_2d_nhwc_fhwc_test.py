# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import torch

from iree.turbine.ops.conv_fwd import conv_2d_nhwc_fhwc

N = 2
H = 16
W = 16
C = 3
F = 1
Hk = 3
Wk = 3

strides = [
    [1, 1],
    [2, 2],
]

dilations = [
    [1, 1],
    [2, 2],
]

device = torch.device("cuda:0") if torch.cuda.is_available() else None


@pytest.mark.parametrize("s", strides)
@pytest.mark.parametrize("d", dilations)
def testCustomConvImplementationEager(s, d):
    gen = torch.Generator(device=device)
    gen.manual_seed(10)
    x = torch.randn([N, H, W, C], generator=gen, dtype=torch.float32, device=device)
    w = torch.randn([F, Hk, Wk, C], generator=gen, dtype=torch.float32, device=device)
    y = conv_2d_nhwc_fhwc(x, w, s, d)
    y_expected = torch.convolution(
        x.permute([0, 3, 1, 2]),
        w.permute([0, 3, 1, 2]),
        None,
        s,
        [0, 0],
        d,
        False,
        [0, 0],
        1,
    ).permute([0, 2, 3, 1])
    assert torch.allclose(
        y, y_expected, rtol=1e-3, atol=1e-3
    ), "Implementation should match."
