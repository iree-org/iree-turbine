# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.kernel.boo.fusion import (
    fused_subgraph,
    extract_fusion_subgraph_modules,
    replace_subgraphs,
    FusionSchema,
    OpFusionSpec,
)
from torch._functorch.aot_autograd import (
    aot_export_joint_simple,
    _aot_export_function,
    aot_module,
    aot_export_module,
)
from torch.fx._symbolic_trace import symbolic_trace

from iree.turbine.kernel.boo.ops import get_custom_graph_op, make_autograd_function


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

    sample_inputs = (torch.randn([B, 3, 32, 32], device=device),)

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
    subgraph_repl = []
    for sg in subgraphs:
        print("printing subgraph:")
        sg.print_readable()
        print("getting fake args from sg:")
        fake_args = tuple(
            [n.meta.get("val") for n in sg.graph.nodes if n.op == "placeholder"]
        )
        # joint_sg = aot_export_joint_simple(sg.forward, args=fake_args, trace_joint=True)
        print("running _aot_export_function:")
        joint_sg, metadata, in_spec, out_spec = _aot_export_function(
            sg.forward,
            fake_args,
            decompositions=None,
        )
        # in_spec, _kw_in_spec = in_spec.children_specs
        print("printing joint subgraph:")
        joint_sg.print_readable()
        fake_args_joint = tuple(
            [n.meta.get("val") for n in joint_sg.graph.nodes if n.op == "placeholder"]
        )
        print(f"{fake_args_joint = }")
        custom_op = make_autograd_function(
            joint_sg, fake_args_joint, num_fwd_outputs=metadata.num_forward_returns
        )
        subgraph_ops.append(custom_op)
        # # custom_op = get_custom_graph_op(sg)
        # # subgraph_ops.append(custom_op)

        # print("making fake module")
        # class FakeMod(torch.nn.Module):
        #     def forward(self, *args):
        #         return custom_op(*args)

        # print("exporting fake module")
        # e = torch.export.export(FakeMod(), args=fake_args)
        single_node_graph = fused_subgraph(
            sg,
            custom_op,
            (n for n in sg.graph.nodes if n.op == "placeholder"),
            num_outputs=metadata.num_forward_returns,
        )
        single_node_graph.print_readable()
        subgraph_repl.append(single_node_graph)

    _ = replace_subgraphs(gm, subgraphs, subgraph_repl)

    print(exported_program)

    converted_module = exported_program.module()

    sample_output = converted_module(*sample_inputs)
    sample_output.sum().backward()


if __name__ == "__main__":
    main()
