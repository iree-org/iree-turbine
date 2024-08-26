# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import sys
import unittest

import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim

# from torch._export.constraints import constrain_as_size, constrain_as_value
from iree.compiler.extras.fx_importer import FxImporter
from shark_turbine.dynamo.passes import turbine_cpu_pass_pipeline
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from torch.fx import (
    GraphModule,
)
from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from iree.compiler.passmanager import (
    PassManager,
)


DEFAULT_COMPILER_FLAGS = (
    # Enable asynchronous calling convention.
    # TODO: Enable async execution mode.
    # "--iree-execution-model=async-external",
    "--iree-input-type=tm_tensor",
)


def import_compiler(gm: GraphModule, example_inputs, decompose_ops=None):
    session = Session()
    session.set_flags(*DEFAULT_COMPILER_FLAGS)
    session.set_flags("--iree-hal-target-backends=llvm-cpu")
    context = session.context
    imp = FxImporter(context=context)
    module = imp.module

    inv = session.invocation()
    # TODO: Should capture diagnostics.
    inv.enable_console_diagnostics()
    inv.import_module(module.operation)

    if decompose_ops is not None:
        gm = make_fx(
            functionalize(gm),
            decomposition_table=get_decompositions(decompose_ops),
        )(*example_inputs)

    gm.print_readable()
    try:
        imp.import_graph_module(gm)
        print(module, file=sys.stderr)
        with context:
            with open("/tmp/module.mlir", "w") as file:
                file.write(str(module))
            pm = PassManager.parse("builtin.module(torch-to-iree)")
            pm.run(module.operation)

    finally:
        print(module, file=sys.stderr)
    module.operation.verify()
    return gm


class DynamicBMM(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight0 = torch.nn.Parameter(torch.rand(n, k))

    def forward(self, inp, *, bias):
        mm = torch.matmul(inp, self.weight0)
        biased = mm + bias
        return {"result": biased}


class DynamicBuiltinOps(torch.nn.Module):
    def forward(self, inp):
        x = inp.size()[1] - inp.size()[2]
        x = x * inp.size()[1] - 34.2
        g = x / 32
        return {"result": g}


class DynamicShapeStridedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        dynamic_shape = [a.size(0), a.size(1), a.size(2)]
        x = torch.ops.aten.empty_strided(
            dynamic_shape, stride=[12, 4, 1]
        )  # Default stride = [12, 4, 1]
        y = x.copy_(a)
        return y


class ImportSmokeTests(unittest.TestCase):
    def testStaticExport(self):
        """
        'tensor.collapse_shape' op expected dimension 0 of collapsed type to be static value of 1
        """
        model = DynamicBMM(12, 19)
        inp_example = torch.rand(1, 2, 12)
        bias_example = torch.rand(19)
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=False,
            assume_static_by_default=True,
            dynamic_shapes={"inp": {1: torch.export.Dim("dim", min=2)}, "bias": None},
        )
        g, guards = f(inp=inp_example, bias=bias_example)
        g = import_compiler(g, [inp_example, bias_example])

    def testStaticExportSameSignatureTrue(self):
        """
        'tensor.collapse_shape' op expected dimension 0 of collapsed type to be static value of 1
        """
        model = DynamicBMM(12, 19)
        inp_example = torch.rand(1, 2, 12)
        bias_example = torch.rand(19)
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=True,
            assume_static_by_default=True,
            dynamic_shapes={"inp": {1: torch.export.Dim("dim", min=2)}, "bias": None},
        )
        g, guards = f(inp=inp_example, bias=bias_example)
        g = import_compiler(g, [inp_example, bias_example])

    # As of torch 2.4, symbolic float support for export is not fully flushed out yet.
    # Skipping test because behavior varies based on torch version.
    # def testStaticExportBuiltinOps(self):
    #     model = DynamicBuiltinOps()
    #     inp_example = torch.rand(1, 2, 12)
    #     f = dynamo.export(
    #         model.forward,
    #         aten_graph=True,
    #         same_signature=True,
    #         assume_static_by_default=True,
    #         dynamic_shapes={"inp": {1: torch.export.Dim("dim", min=2)}},
    #     )
    #     g, guards = f(inp=inp_example)
    #     g = import_compiler(g, [inp_example])

    @unittest.expectedFailure
    def testDynamicShapeStrided(self):
        """
        Regardless of default stride=[12, 4, 1] provided, we get the following error.
         failed to legalize operation 'torch.constant.int'
         By Dumping IR, you get the following.
         /tmp/module.mlir:7:10: error: 'tensor.collapse_shape' op expected dimension 0 of collapsed type to be static value of 1
         %2 = torch.aten.view %arg0, %1 : !torch.vtensor<[1,?,12],f32>, !torch.list<int> -> !torch.vtensor<[?,12],f32>
        """
        model = DynamicShapeStridedModule()
        # inp_example = torch.rand(5, 7, 9)
        inp_example = torch.randn(2, 3, 4)  # input for default stride
        f = dynamo.export(
            model.forward,
            aten_graph=True,
            same_signature=True,
            assume_static_by_default=True,
            dynamic_shapes={"inp": {1: torch.export.Dim("dim", min=2)}},
        )
        g, guards = f(a=inp_example)
        g = import_compiler(g, [inp_example])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
