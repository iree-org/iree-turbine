# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from pathlib import Path

import torch

from iree.turbine.kernel.boo.ops import get_custom_graph_op
from iree.turbine.kernel.boo.runtime.cache import OpCacheFiles


class SampleModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class LongFusionSample(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, iter: int = 10):
        super().__init__()
        self.iter = iter
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features
        )
        self.act = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        y = self.linear(x)
        for _ in range(self.iter):
            y = self.act(y)
        return y


def test_long_path_name():
    m = LongFusionSample(16, 32, 10)
    x = torch.ones([3, 3, 16, 16])

    with torch.no_grad():
        exported = torch.export.export(m, args=(x,))

    gm = exported.graph_module

    # Get a graph op (note, model params are pytree flattened as args).
    graph_op = get_custom_graph_op(gm)
    assert len(graph_op._qualified_op_name) < 256 - len(
        ".mlir"
    ), f"Name: '{graph_op._qualified_op_name}' is too long."


@pytest.mark.parametrize("inplace_convert", [True, False])
def test_forward_only_op(inplace_convert: bool, boo_cache_dir: Path):
    m = SampleModule(16, 32)
    x = torch.ones([3, 3, 16, 16])

    # Export a graph.
    exported = torch.export.export(m, args=(x,))
    gm = exported.graph_module

    # Get a graph op (note, model params are pytree flattened as args).
    graph_op = get_custom_graph_op(gm, inplace_convert=inplace_convert)

    # Apply the graph op.
    y = graph_op(m.linear.weight, m.linear.bias, x)
    assert list(y.shape) == [3, 3, 16, 32]
    # Since we exported with static dims, applying to an input with a different shape should throw an error.
    new_x = torch.ones([4, 3, 32, 16])
    with pytest.raises(
        ValueError, match=r"INVALID_ARGUMENT; tensor shape dimension 0 mismatch"
    ):
        new_y = graph_op(m.linear.weight, m.linear.bias, new_x)

    # Verify timeline progressed after failure.
    y = graph_op(m.linear.weight, m.linear.bias, x)

    # Check op name.
    op_name = graph_op._qualified_op_name.split("::")[-1]
    if inplace_convert:
        assert "inplace" in op_name

    # Verify caching.
    cache_subdir_names = [p.name for p in boo_cache_dir.glob("*/")]
    assert op_name in cache_subdir_names

    op_cache_files = OpCacheFiles(op_name)

    assert op_cache_files.mlir_path.is_file()

    # Check in-place indicator ops exist in the torch IR.
    mlir_asm = op_cache_files.mlir_path.read_text()
    assert ("torch.overwrite.tensor.contents" in mlir_asm) == (
        inplace_convert
    ), "Must have overwrite op iff inplace_convert=True."
