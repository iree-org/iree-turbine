# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires cuda"
            ),
        ),
    ],
)
def test_generic_backend_inference_only(device: torch.device):
    """When setup for backward, the forward function returns partial results to be cached for the backward pass.
    It's possible that returning intermediate results could cause some issues in the compiler.
    We make this test inference-only to compare against the results of the inference + bwd test below."""

    @torch.compile(backend="iree_turbine")
    def conv_relu_mul(x, y, z, w):
        conv = torch.conv2d(x, y, z)
        relu = torch.nn.functional.relu(conv)
        return relu * w

    x = torch.ones([2, 3, 16, 16], dtype=torch.float32, device=device)
    y = torch.ones([2, 3, 1, 1], dtype=torch.float32, device=device)
    z = torch.tensor([-2.5, -3.5], dtype=torch.float32, device=device)
    w = 0.2 * torch.ones([2, 2, 16, 16], dtype=torch.float32, device=device)

    output = conv_relu_mul(x, y, z, w)

    atol = 1e-6
    max_err = torch.max(torch.abs(output[:, 0, :, :] - (0.5 * 0.2))).item()
    max_err = max(max_err, torch.max(torch.abs(output[:, 1, :, :])).item())
    # Note: rel error is not super useful to check for forward since:
    assert max_err <= atol, f"Forward numerics failure: {max_err=} for {atol=}."


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=(
                pytest.mark.xfail(reason="Forward numerics failure requires triage."),
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="requires cuda"
                ),
            ),
        ),
    ],
)
def test_generic_backend_backward(device: torch.device):
    @torch.compile(backend="iree_turbine")
    def conv_relu_mul(x, y, z, w):
        conv = torch.conv2d(x, y, z)
        relu = torch.nn.functional.relu(conv)
        return relu * w

    x = torch.ones(
        [2, 3, 16, 16], dtype=torch.float32, device=device, requires_grad=True
    )
    y = torch.ones([2, 3, 1, 1], dtype=torch.float32, device=device, requires_grad=True)
    z = torch.tensor(
        [-2.5, -3.5], dtype=torch.float32, device=device, requires_grad=True
    )
    w = 0.2 * torch.ones(
        [2, 2, 16, 16], dtype=torch.float32, device=device, requires_grad=True
    )
    x.retain_grad()
    y.retain_grad()
    z.retain_grad()
    w.retain_grad()

    output = conv_relu_mul(x, y, z, w)
    output.sum().backward()

    # dL/dL = 1

    # dL/doutput = ones_like(output)

    # dL/drelu = w

    # dL/dw = relu = output/0.2 = 0.5 or 0 (if output channel = 0 or 1 resp.).

    # dL/dconv = relu^*(w)
    # dL/dconv[i,j,k,l] = w[i,0,k,l] = 0.2 if j = 0 else 0

    # dL/dz[j] = sum_(i,k,l) (dL/dconv)[i,j,k,l] = 0.2*2*16*16 if j = 0 else 0

    # dL/dy = wrw_conv(dLdconv, x) = conv_over_batch_dim(dLdconv, x)
    # dL/dy[f,c,kh,kw] = 2*16*16*dLdconv[0, f, 0, 0] = 2*16*16*0.2 if f = 0 else 0.0

    # dL/dx = bwd_conv(dLdconv, y) = conv_over_output_channel_dim(dLdconv, y)
    # dL/dx[i,j,k,l] = dLdconv[i,0,k,l] + dLdconv[i,1,k,l] = 0.2

    result_expectations = (
        (output[:, 0, :, :], 0.5 * 0.2, "forward channel 0"),
        (output[:, 1, :, :], 0.0, "forward channel 1"),
        (w.grad[:, 0, :, :], 0.5, "w.grad channel 0"),
        (w.grad[:, 1, :, :], 0.0, "w.grad channel 1"),
        (z.grad[0], 0.2 * 2 * 16 * 16, "z.grad channel 0"),
        (z.grad[1], 0.0, "z.grad channel 1"),
        (y.grad[0, :, :, :], 0.2 * 2 * 16 * 16, "y.grad channel 0"),
        (y.grad[1, :, :, :], 0.0, "y.grad channel 1"),
        (x.grad, 0.2, "x.grad"),
    )

    atol = 1e-8
    rtol = 1e-5
    abs_err = lambda tensor, expected_value: torch.max(
        torch.abs(tensor - expected_value)
    ).item()
    compare = lambda err, expected_value: err <= atol + rtol * expected_value
    message = ""
    for tensor, expected_value, label in result_expectations:
        err = abs_err(tensor, expected_value)
        success = compare(err, expected_value)
        if not success:
            message += f"Failed numerics for {label}: {err=}.\n"

    assert message == "", message
