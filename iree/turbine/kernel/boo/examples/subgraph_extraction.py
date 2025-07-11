# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import torch
from torch.profiler import profile, ProfilerActivity
from functools import partial

from iree.turbine.dynamo.backends import boo
from iree.turbine.kernel.boo.fusion import (
    FusionSchema,
    OpFusionSpec,
)


class SampleModel(torch.nn.Module):
    def __init__(self, pixel_width: int = 32, pixel_height: int = 32):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            torch.nn.Linear(
                in_features=(pixel_height - 2) * (pixel_width - 2),
                out_features=(pixel_width) * (pixel_height),
            ),
            torch.nn.Unflatten(dim=-1, unflattened_size=[pixel_width, pixel_height]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        return x2


def main(print_parameters: bool, trace_path: str):
    def dump_profile(profiler: profile):
        profiler.export_chrome_trace(trace_path)

    def profiler_ctx(enabled: bool = False) -> profile:
        if not enabled:
            return contextlib.nullcontext()
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=partial(dump_profile),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = SampleModel().to(device=device)

    B = 4

    sample_inputs = (torch.randn([B, 3, 32, 32], device=device),)

    # TODO: identify a default FusionSchema to use.
    # This schema indicates that we are always offloading `conv2d` and `linear` ops to IREE.
    # If any convolution is followed by a relu, it will be fused into the IREE's convolution kernel.
    # If any linear is preceeded or followed by view ops, those will be fused into the linear op.
    schema: FusionSchema = {
        torch.ops.aten.convolution.default: OpFusionSpec(
            recursive=True, producers=(), consumers=(torch.ops.aten.relu.default,)
        ),
        torch.ops.aten.addmm.default: OpFusionSpec(
            recursive=True,
            producers=(torch.ops.aten.view.default,),
            consumers=(torch.ops.aten.view.default,),
        ),
    }

    converted_module = torch.compile(m, backend=boo.backend(fusion_schema=schema))

    # warmup
    sample_output = converted_module(*sample_inputs)
    sample_output.sum().backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profiler_ctx(trace_path != ""):
        sample_output = converted_module(*sample_inputs)
        sample_output.sum().backward()

    if print_parameters:
        for name, param in m.named_parameters():
            print(f"parameter {name}:\n{param.data = }\n{param.grad = }")

    # opt_conv = torch.compile(converted_module)

    # output = opt_conv(*sample_inputs)
    # output.sum().backward()


if __name__ == "__main__":
    main(False, "")
