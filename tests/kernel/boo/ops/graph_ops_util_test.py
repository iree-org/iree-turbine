# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from pathlib import Path

import torch

from iree.turbine.kernel.boo.ops.graph import (
    get_custom_graph_op,
    GraphSchema,
    BoundaryClassification,
    BoundaryValue,
)
from iree.turbine.support.ir_imports import Context, Module, SymbolTable


class SampleModule(torch.nn.Module):
    def forward(self, x, y, a):
        """
        x : user input
        y : user input
        a : unused input

        output[0] : unique output
        output[1] : None output
        output[2] : no-op
        output[3] : repeat of output[0]
        output[4] : unique output
        """
        z = torch.matmul(x, y)
        w = x / torch.norm(y, 2.0)
        return z, None, x, z, w


def test_graph_boundary():
    """Tests that we get an appropriate GraphSchema for the sample module.

    This also tests that the composition rules work as expected.
    """
    ep = torch.export.export(
        SampleModule(), (torch.randn([3, 4]), torch.randn([4, 1]), torch.randn([1, 2]))
    )
    ep.graph_module.print_readable(include_stride=True)
    schema = GraphSchema.from_gm(ep.graph_module)
    placeholders = schema.placeholders
    outputs = schema.outputs
    assert len(placeholders) == 3, "Expected 3 inputs."
    assert len(outputs) == 5, "Expected 5 inputs."
    x_b_v, y_b_v, a_b_v = placeholders
    z_b_v, None_b_v, x_noop_b_v, z_1_b_v, w_b_v = outputs
    assert x_b_v.classification == BoundaryClassification.USER_INPUT
    assert x_b_v.index == 0
    assert y_b_v.classification == BoundaryClassification.USER_INPUT
    assert y_b_v.index == 1
    assert a_b_v.classification == BoundaryClassification.UNUSED_INPUT
    assert a_b_v.index == 2
    assert z_b_v.classification == BoundaryClassification.UNIQUE_OUTPUT
    assert z_b_v.index == 0
    assert None_b_v.classification == BoundaryClassification.NONE_OUTPUT
    assert None_b_v.index == 1
    assert x_noop_b_v.classification == BoundaryClassification.NO_OP_OUTPUT
    assert x_noop_b_v.index == 2
    assert z_1_b_v.classification == BoundaryClassification.REPEATED_OUTPUT
    assert z_1_b_v.index == 3
    assert w_b_v.classification == BoundaryClassification.UNIQUE_OUTPUT
    assert w_b_v.index == 4

    inner_pl = [x_b_v, y_b_v]
    inner_outputs = [z_b_v, w_b_v]

    # The input composition rule should
    all_input_values = list(inp.value for inp in placeholders)
    inner_input_values = list(inp.value for inp in inner_pl)
    inner_output_values = list(out.value for out in inner_outputs)

    # Inner inputs should be grabbed from original input values.
    for pl in inner_pl:
        assert pl.composition_rule(all_input_values, []) == pl.value

    # We should recover each original output by inner inputs and outputs.
    for o in outputs:
        assert o.composition_rule(inner_input_values, inner_output_values) == o.value
