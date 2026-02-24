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
    GraphTransformation,
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


class TransformTestModule(torch.nn.Module):
    """forward(x, y, unused) -> (z, z, w) where z=x+y, w=x*y

    Boundary classifications:
    - x: USER_INPUT, y: USER_INPUT, unused: UNUSED_INPUT
    - z (first): UNIQUE_OUTPUT, z (second): REPEATED_OUTPUT, w: UNIQUE_OUTPUT
    """

    def forward(self, x, y, unused):
        z = x + y
        w = x * y
        return z, z, w


def test_get_inner_graph_transformation():
    """Tests that get_inner_graph_transformation produces a correct transformation.

    Verifies the structure of mod_gm, correctness of input_mod and output_mod,
    and end-to-end reconstruction of original outputs.
    """
    x = torch.randn([3, 4])
    y = torch.randn([3, 4])
    unused = torch.randn([2, 2])

    ep = torch.export.export(TransformTestModule(), (x, y, unused))
    schema = GraphSchema.from_gm(ep.graph_module)
    t = schema.get_inner_graph_transformation()

    # Verify src_gm is the original graph module.
    assert t.src_gm is ep.graph_module

    # Verify mod_gm has only inner inputs/outputs.
    mod_pls = list(t.mod_gm.graph.find_nodes(op="placeholder"))
    mod_out = t.mod_gm.graph.output_node()
    assert len(mod_pls) == 2, "mod_gm should have 2 inputs (x, y)"
    assert len(mod_out.args[0]) == 2, "mod_gm should have 2 outputs (z, w)"

    # Verify input_mod extracts the correct inner inputs.
    inner_inputs = t.input_mod([x, y, unused])
    assert len(inner_inputs) == 2
    assert torch.equal(inner_inputs[0], x)
    assert torch.equal(inner_inputs[1], y)

    # Run mod_gm and verify output_mod reconstructs all original outputs.
    mod_outputs = list(t.mod_gm(*inner_inputs))
    reconstructed = t.output_mod([x, y, unused], mod_outputs)

    expected = list(ep.graph_module(x, y, unused))
    assert len(reconstructed) == len(expected) == 3
    for i in range(3):
        assert torch.allclose(reconstructed[i], expected[i])


def _make_empty_graph_module():
    """Helper to create a minimal GraphModule for identity checks."""
    g = torch.fx.Graph()
    g.output(None)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def test_graph_transformation_composition():
    """Tests that __mul__ correctly composes two GraphTransformations.

    Sets up:
        t_other: gm_a -> gm_b
            input_mod: [a, b] -> [a + b]
            output_mod: (inputs, outputs) -> outputs + [inputs[1]]
        t_self: gm_b -> gm_c
            input_mod: [x] -> [x * 2]
            output_mod: (inputs, outputs) -> [outputs[0] + inputs[0]]

    Composed (t_self * t_other): gm_a -> gm_c
        input_mod: [a, b] -> [(a+b) * 2]
        output_mod: ([a, b], [c]) -> [c + (a+b), b]
    """
    gm_a, gm_b, gm_c = (
        _make_empty_graph_module(),
        _make_empty_graph_module(),
        _make_empty_graph_module(),
    )

    t_other = GraphTransformation(
        src_gm=gm_a,
        mod_gm=gm_b,
        input_mod=lambda inputs: [inputs[0] + inputs[1]],
        output_mod=lambda inputs, outputs: outputs + [inputs[1]],
    )
    t_self = GraphTransformation(
        src_gm=gm_b,
        mod_gm=gm_c,
        input_mod=lambda inputs: [inputs[0] * 2],
        output_mod=lambda inputs, outputs: [outputs[0] + inputs[0]],
    )

    composite = t_self * t_other
    assert composite.src_gm is gm_a
    assert composite.mod_gm is gm_c

    # Verify composed input_mod.
    assert composite.input_mod([3.0, 5.0]) == [16.0]

    # Verify composed output_mod.
    assert composite.output_mod([3.0, 5.0], [10.0]) == [18.0, 5.0]


def test_graph_transformation_composition_mismatch():
    """Tests that composing transformations with mismatched graph modules raises."""
    gm_a, gm_b, gm_c = (
        _make_empty_graph_module(),
        _make_empty_graph_module(),
        _make_empty_graph_module(),
    )

    t1 = GraphTransformation(
        src_gm=gm_a,
        mod_gm=gm_b,
        input_mod=lambda x: x,
        output_mod=lambda x, y: y,
    )
    t2 = GraphTransformation(
        src_gm=gm_a,  # Should be gm_b to match t1.mod_gm
        mod_gm=gm_c,
        input_mod=lambda x: x,
        output_mod=lambda x, y: y,
    )

    with pytest.raises(AssertionError, match="Mismatched"):
        _ = t2 * t1


def test_get_inner_graph_transformation_with_none_and_noop():
    """Tests get_inner_graph_transformation on a graph with None and no-op outputs.

    SampleModule: forward(x, y, a) -> (z, None, x, z, w)
    Inner inputs: x, y  (a is UNUSED_INPUT)
    Inner outputs: z, w  (None is NONE_OUTPUT, x is NO_OP, z repeat is REPEATED)
    """
    x = torch.randn([3, 4])
    y = torch.randn([4, 1])
    a = torch.randn([1, 2])

    ep = torch.export.export(SampleModule(), (x, y, a))
    schema = GraphSchema.from_gm(ep.graph_module)
    t = schema.get_inner_graph_transformation()

    assert t.src_gm is ep.graph_module

    # mod_gm should have 2 inner inputs (x, y) and 2 inner outputs (z, w).
    mod_pls = list(t.mod_gm.graph.find_nodes(op="placeholder"))
    mod_out = t.mod_gm.graph.output_node()
    assert len(mod_pls) == 2, "mod_gm should have 2 inputs (x, y)"
    assert len(mod_out.args[0]) == 2, "mod_gm should have 2 outputs (z, w)"

    # Verify input_mod extracts the correct inner inputs.
    inner_inputs = t.input_mod([x, y, a])
    assert len(inner_inputs) == 2
    assert torch.equal(inner_inputs[0], x)
    assert torch.equal(inner_inputs[1], y)

    # Run mod_gm and verify output_mod reconstructs all 5 original outputs.
    mod_outputs = list(t.mod_gm(*inner_inputs))
    reconstructed = t.output_mod([x, y, a], mod_outputs)

    expected = list(ep.graph_module(x, y, a))
    for r, exp in zip(reconstructed, expected, strict=True):
        if isinstance(r, torch.Tensor):
            assert isinstance(exp, torch.Tensor), "Expected both to be tensors."
            assert torch.equal(r, exp), "Expected tensors to be exactly equal."
            continue
        assert r == exp, "Non-tensor values must match."
