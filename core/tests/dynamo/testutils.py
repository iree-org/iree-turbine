# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
from typing import List

from shark_turbine.importers.fx_importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from torch.fx import (
    GraphModule,
)


def create_backend(decompose_ops: List[torch._ops.OpOverloadPacket] = None):
    imp = FxImporter()

    def import_compiler(gm: GraphModule, example_inputs):
        if decompose_ops is not None:
            gm = make_fx(
                functionalize(gm),
                decomposition_table=get_decompositions(decompose_ops),
            )(*example_inputs)

        gm.print_readable()
        try:
            imp.import_graph_module(gm)
        finally:
            print(imp.module)
        imp.module.operation.verify()
        return gm

    backend = import_compiler
    backend = aot_autograd(fw_compiler=backend)
    return backend
