# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import nn
from torch.testing import assert_close

from iree.turbine.kernel.wave.layers.linear import WaveLinear
from ..common.utils import require_e2e


@require_e2e
def testLinearNoBatch():
    # Linear layer parameters
    torch.manual_seed(1)
    input_len = 32
    in_features = 64
    out_features = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Setup reference linear layer and wave linear layer.
    ref_linear = nn.Linear(in_features, out_features, device=device, dtype=dtype)
    wave_linear = WaveLinear(in_features, out_features, device=device, dtype=dtype)

    # Copy data from reference torch
    with torch.no_grad():
        wave_linear.weight.copy_(ref_linear.weight.data)
        wave_linear.bias.copy_(ref_linear.bias.data)

    # Run and compare output
    test_inputs = torch.randn(input_len, in_features, dtype=dtype, device=device)
    ref_output = ref_linear.forward(test_inputs)
    wave_output = wave_linear.forward(test_inputs)

    assert_close(
        wave_output,
        ref_output,
        atol=1e-3,
        rtol=1e-3,
        check_dtype=False,
        check_device=False,
    )


@require_e2e
def testLinear1DBatchNoBias():
    # Linear layer parameters
    torch.manual_seed(1)
    batch = 16
    input_len = 32
    in_features = 64
    out_features = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Setup reference linear layer and wave linear layer.
    ref_linear = nn.Linear(
        in_features, out_features, device=device, dtype=dtype, bias=False
    )
    wave_linear = WaveLinear(
        in_features, out_features, device=device, dtype=dtype, bias=False
    )

    # Copy data from reference torch
    with torch.no_grad():
        wave_linear.weight.copy_(ref_linear.weight.data)

    # Run and compare output
    test_inputs = torch.randn(batch, input_len, in_features, dtype=dtype, device=device)
    ref_output = ref_linear.forward(test_inputs)
    wave_output = wave_linear.forward(test_inputs)

    assert_close(
        wave_output,
        ref_output,
        atol=1e-3,
        rtol=1e-3,
        check_dtype=False,
        check_device=False,
    )


@require_e2e
def testLinear1DBatch():
    # Linear layer parameters
    torch.manual_seed(1)
    batch = 16
    input_len = 32
    in_features = 64
    out_features = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Setup reference linear layer and wave linear layer.
    ref_linear = nn.Linear(in_features, out_features, device=device, dtype=dtype)
    wave_linear = WaveLinear(in_features, out_features, device=device, dtype=dtype)

    # Copy data from reference torch
    with torch.no_grad():
        wave_linear.weight.copy_(ref_linear.weight.data)
        wave_linear.bias.copy_(ref_linear.bias.data)

    # Run and compare output
    test_inputs = torch.randn(batch, input_len, in_features, dtype=dtype, device=device)
    ref_output = ref_linear.forward(test_inputs)
    wave_output = wave_linear.forward(test_inputs)

    assert_close(
        wave_output,
        ref_output,
        atol=1e-3,
        rtol=1e-3,
        check_dtype=False,
        check_device=False,
    )


@require_e2e
def testLinear3DBatch():
    # Linear layer parameters
    torch.manual_seed(1)
    batch = [2, 4, 8]
    input_len = 32
    in_features = 64
    out_features = 128
    device = torch.device("cuda:0")
    dtype = torch.float16

    # Setup reference linear layer and wave linear layer.
    ref_linear = nn.Linear(in_features, out_features, device=device, dtype=dtype)
    wave_linear = WaveLinear(in_features, out_features, device=device, dtype=dtype)

    # Copy data from reference torch
    with torch.no_grad():
        wave_linear.weight.copy_(ref_linear.weight.data)
        wave_linear.bias.copy_(ref_linear.bias.data)

    # Run and compare output
    test_inputs = torch.randn(
        batch[0], batch[1], batch[2], input_len, in_features, dtype=dtype, device=device
    )
    ref_output = ref_linear.forward(test_inputs)
    wave_output = wave_linear.forward(test_inputs)

    assert_close(
        wave_output,
        ref_output,
        atol=1e-3,
        rtol=1e-3,
        check_dtype=False,
        check_device=False,
    )
