# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.kernel.boo.fusion import (
    extract_fusion_subgraph_modules,
    FusionSchema,
    OpFusionSpec,
)

from iree.turbine.kernel.boo.ops import get_custom_graph_op


class SampleModel(torch.nn.Module):
    def __init__(self, pixel_width: int = 256, pixel_height: int = 256):
        super().__init__()
        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
            torch.nn.ReLU(),
        )
        # spatial output shape of layer 0 = output shape layer 1 = (h - 1) - (k - 1) + 1 = h - 2
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

    sample_inputs = (torch.randn([B, 3, 256, 256], device=device),)

    exported_program = torch.export.export(m, args=sample_inputs)

    gm = exported_program.graph_module

    gm.print_readable()

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

    subgraphs, _ = extract_fusion_subgraph_modules(gm, schema)
    subgraph_ops = []
    for sg in subgraphs:
        sg.print_readable()
        subgraph_ops.append(get_custom_graph_op(sg))


if __name__ == "__main__":
    main()
