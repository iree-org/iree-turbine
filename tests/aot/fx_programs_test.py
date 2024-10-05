# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import tempfile

import pytest
import torch

from iree.turbine.aot import (
    FxPrograms,
    FxProgramsBuilder,
)


def test_save_load_dynamic_shapes():
    if torch.__version__ < "2.3.0.dev1":
        pytest.skip("Unsupported PyTorch version")

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

    example_args = (torch.randn(32, 64), torch.randn(32, 128))

    # Create a dynamic batch size
    batch = torch.export.Dim("batch")
    # Specify that the first dimension of each input is that batch size
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    fxb = FxProgramsBuilder(M())

    @fxb.export_program(args=example_args, dynamic_shapes=dynamic_shapes)
    def dynamic_batch(module: M, x1, x2):
        return module.forward(x1, x2)

    @fxb.export_program(args=example_args)
    def bs32(module: M, x1, x2):
        return module.forward(x1, x2)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "branchy.json"
        dedup_count = fxb.save(p)
        assert dedup_count == 5  # Two sets of weights/bias and one constant
        new_programs = FxPrograms.load(p)

    prog_0 = new_programs.programs["dynamic_batch"]
    prog_1 = new_programs.programs["bs32"]

    for key, value_0 in prog_0.state_dict.items():
        value_1 = prog_1.state_dict[key]
        assert value_0 is value_1, f"State dict item {key} was not aliased on load"

    for key, value_0 in prog_0.constants.items():
        value_1 = prog_1.constants[key]
        assert value_0 is value_1, f"Constant item {key} was not aliased on load"
