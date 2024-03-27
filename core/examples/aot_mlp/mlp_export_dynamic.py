# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This sample builds a dynamic shape version of the MLP with
# a dynamic batch dimension. It uses the advanced, low-level
# API because we don't have dynamic shapes available in the
# simple API yet.

import torch
import torch.nn as nn

import shark_turbine.aot as aot


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(8, 8, bias=True)
        self.layer1 = nn.Linear(8, 4, bias=True)
        self.layer2 = nn.Linear(4, 2, bias=True)
        self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        x = torch.sigmoid(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return x


model = MLP()


class CompiledMLP(aot.CompiledModule):
    params = aot.export_parameters(model)

    def main(self, x=aot.AbstractTensor(None, 97, 8, dtype=torch.float32)):
        return aot.jittable(model.forward)(
            x,
            constraints=[
                x.dynamic_dim(0),
            ],
        )


batch = torch.export.Dim("batch")
exported = aot.export(
    model,
    args=(torch.empty([2, 97, 8], dtype=torch.float32),),
    dynamic_shapes={"x": {0: batch}},
)
# Note that dynamic Torch IR is created below.
exported.print_readable()


# TODO: Enable once version roll to ToT torch-mlir with dynamic view
# op legalization fixes.
# compiled_binary = exported.compile(save_to=None)
# def infer():
#     import numpy as np
#     import iree.runtime as rt

#     config = rt.Config("local-task")
#     vmm = rt.load_vm_module(
#         rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
#         config,
#     )
#     x = np.random.rand(10, 97, 8).astype(np.float32)
#     y = vmm.main(x)
#     print(y.to_host())
# infer()
