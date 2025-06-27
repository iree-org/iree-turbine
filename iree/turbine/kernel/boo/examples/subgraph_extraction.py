# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from iree.turbine.kernel.boo.fusion import (
    fusion_transform,
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

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        return x2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = SampleModel().to(device=device)

    B = 4

    sample_inputs = (torch.randn([B, 3, 32, 32], device=device),)

    # sample_output = m(*sample_inputs)
    # sample_output.sum().backward()
    # for name, param in m.named_parameters():
    #     print(f'parameter {name}:\n{param.data = }\n{param.grad = }')

    # import sys
    # sys.exit(0)

    schema: FusionSchema = {
        torch.ops.aten.conv2d.default: OpFusionSpec(
            recursive=True, producers=(), consumers=(torch.ops.aten.relu.default,)
        ),
        torch.ops.aten.linear.default: OpFusionSpec(
            recursive=True,
            producers=(torch.ops.aten.view.default,),
            consumers=(torch.ops.aten.view.default,),
        ),
    }

    converted_module = fusion_transform(m, sample_inputs, fusion_schema=schema)

    sample_output = converted_module(*sample_inputs)
    sample_output.sum().backward()
    for name, param in m.named_parameters():
        print(f"parameter {name}:\n{param.data = }\n{param.grad = }")


if __name__ == "__main__":
    main()
