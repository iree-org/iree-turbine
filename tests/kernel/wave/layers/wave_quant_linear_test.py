# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import nn
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    require_cdna3,
)
from iree.turbine.kernel.wave.layers.quant_linear import WaveQuantLinear
from iree.turbine.kernel.wave.utils.general_utils import torch_dtype_range
from ..common.utils import require_e2e


class RefQuantLinear(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        quant_params,
        device,
        dtype=torch.float16,
        bias=True,
    ):
        super().__init__()
        self.device = device
        self.bias = bias
        self.dtype = dtype
        self.linear = nn.Linear(
            in_channels, out_channels, device=self.device, bias=bias
        )
        self.qdtype = quant_params["qdtype"]
        weight_scale = (
            quant_params["weight_scale"]
            .clone()
            .detach()
            .view(quant_params["weight_scale_shape"])
            .to(self.device)
        )
        input_scale = (
            quant_params["input_scale"]
            .clone()
            .detach()
            .view(quant_params["input_scale_shape"])
            .to(self.device)
        )
        # Get limits of representable range for the quant data type.
        self.clamp_min = torch.tensor(
            torch_dtype_range(self.qdtype)[0], dtype=dtype, device=self.device
        )
        self.clamp_max = torch.tensor(
            torch_dtype_range(self.qdtype)[1], dtype=dtype, device=self.device
        )
        quant_weight = self.quantize_tensor(
            self.linear.weight.detach(), weight_scale
        ).to(self.qdtype)
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("quant_weight", quant_weight)
        self.register_buffer("weight_scale", weight_scale)

    def forward(self, x):
        quant_input = self.quantize_tensor(x, self.input_scale).to(self.qdtype)
        quant_output = torch.nn.functional.linear(
            quant_input.to(torch.float32), self.quant_weight.to(torch.float32), None
        ).to(
            torch.float32
        )  # Convert inputs to FP32 to avoid F.linear quantizing the output to int8
        output = self.dequantize_tensor(
            quant_output,
            (self.weight_scale * self.input_scale).view(
                [1] * (quant_output.ndim - 1)
                + [(self.weight_scale * self.input_scale).nelement()]
            ),
        )
        if self.bias:
            output += self.linear.bias
        return output

    def quantize_tensor(self, tensor: torch.Tensor, scale: torch.Tensor):
        quant_tensor = (
            torch.clamp((tensor / scale), self.clamp_min, self.clamp_max)
            .to(self.qdtype)
            .to(self.dtype)
        )
        return quant_tensor

    def dequantize_tensor(self, tensor: torch.Tensor, scale: torch.Tensor):
        return tensor * scale


@require_e2e
@require_cdna3
def testQLinearPerTensor1DBatchNoBias():
    torch.manual_seed(1)
    batch = 16
    input_len = 32
    in_features = 64
    out_features = 128
    device = torch.device("cuda:0")
    dtype = torch.float16
    quant_params = {
        "weight_scale": torch.rand(1),
        "weight_scale_shape": [1],
        "input_scale": torch.rand(1),
        "input_scale_shape": [1],
        "qdtype": torch.float8_e4m3fnuz,
    }
    # Setup reference linear layer and wave linear layer.
    ref_linear = RefQuantLinear(
        in_features,
        out_features,
        quant_params=quant_params,
        device=device,
        bias=False,
    )
    wave_linear = WaveQuantLinear(
        in_features,
        out_features,
        quant_params=quant_params,
        device=device,
        dtype=dtype,
        bias=False,
    )
    # Copy data from reference torch
    with torch.no_grad():
        wave_linear.weight.copy_(ref_linear.linear.weight.data)

    # Run and compare output
    test_inputs = torch.randn(batch, input_len, in_features, dtype=dtype, device=device)
    ref_output = ref_linear.forward(test_inputs)
    wave_output = wave_linear.forward(test_inputs)
    absmax_error = torch.abs(torch.max(wave_output - ref_output))
    rmse_error = torch.sqrt(torch.mean((wave_output - ref_output) ** 2)).item()
    assert absmax_error < 1e-1, "absmax is not less than the threshold"
    assert rmse_error < 1e-2, "RMSE is not less than the threshold"
