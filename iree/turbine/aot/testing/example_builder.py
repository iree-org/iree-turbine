# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test iree.build file for exercising turbine_generate.

Since we can only do in-process testing of real modules (not dynamically loaded)
across process boundaries, this builder must live in the source tree vs the
tests tree.
"""

import os

from iree.build import compile, entrypoint, iree_build_main
from iree.turbine.aot.build_actions import *
from iree.turbine.aot import (
    ExportOutput,
    FxProgramsBuilder,
    export,
    externalize_module_parameters,
)


def export_simple_model(batch_size: int | None = None) -> ExportOutput:
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU())
            self.branch2 = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example_bs = 2 if batch_size is None else batch_size
    example_args = (torch.randn(example_bs, 64), torch.randn(example_bs, 128))

    # Create a dynamic batch size
    if batch_size is None:
        batch = torch.export.Dim("batch")
        # Specify that the first dimension of each input is that batch size
        dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}
    else:
        dynamic_shapes = {}

    module = M()
    externalize_module_parameters(module)
    fxb = FxProgramsBuilder(module)
    print(f"  [{os.getpid()}] Compiling with dynamic shapes: {dynamic_shapes}")

    @fxb.export_program(args=example_args, dynamic_shapes=dynamic_shapes)
    def dynamic_batch(module: M, x1, x2):
        return module.forward(x1, x2)

    return export(fxb)


@entrypoint(description="Builds an awesome pipeline")
def pipe():
    print(f"Main pid: {os.getpid()}")
    results = []
    for i in range(3):
        turbine_generate(
            export_simple_model,
            batch_size=None if i == 0 else i * 10,
            name=f"import_stage{i}",
            out_of_process=i > 0,
        )
        results.extend(
            compile(
                name=f"stage{i}",
                source=f"import_stage{i}.mlir",
            )
        )
    return results


if __name__ == "__main__":
    iree_build_main()
