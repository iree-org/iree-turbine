# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
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
example_x = torch.empty(97, 8, dtype=torch.float32)
exported = aot.export(model, example_x)
exported.print_readable()
compiled_binary = exported.compile(save_to=None)


def infer():
    import numpy as np
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    x = np.random.rand(97, 8).astype(np.float32)
    y = vmm.main(x)
    print(y.to_host())


class ModelTest(unittest.TestCase):
    def testMLPExportSimple(selfs):
        infer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
