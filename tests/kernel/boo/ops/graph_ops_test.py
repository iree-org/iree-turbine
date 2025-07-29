# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import tempfile
import pytest

from pathlib import Path

import torch

from iree.turbine.kernel.boo.ops import get_custom_graph_op
from iree.turbine.kernel.boo.runtime import set_cache_dir, LaunchableRuntimeCache


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


class TestGraphOps:
    @pytest.fixture(autouse=True)
    def setUp(self):
        LaunchableRuntimeCache.clear()
        LaunchableRuntimeCache.set_cache_limit(0)

    def testLongPathName(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))

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

    def testForwardOnlyOp(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))

            m = SampleModule(16, 32)
            x = torch.ones([3, 3, 16, 16])

            # Export a graph.
            exported = torch.export.export(m, args=(x,))
            gm = exported.graph_module

            # Get a graph op (note, model params are pytree flattened as args).
            graph_op = get_custom_graph_op(gm)

            # Apply the graph op.
            y = graph_op(m.linear.weight, m.linear.bias, x)
            assert list(y.shape) == [3, 3, 16, 32]
            # Since we exported with static dims, applying to an input with a different shape should throw an error.
            new_x = torch.ones([4, 3, 32, 16])
            with pytest.raises(
                ValueError, match=r"INVALID_ARGUMENT; tensor shape dimension 0 mismatch"
            ):
                new_y = graph_op(m.linear.weight, m.linear.bias, new_x)

            # Verify caching.
            op_name = graph_op._qualified_op_name.split("::")[-1]
            cache_subdir_names = [p.name for p in Path(td).glob("*/")]
            expected_dir_name_0 = (
                op_name + "_32x16xfloat32_32xfloat32_3x3x16x16xfloat32"
            )
            assert expected_dir_name_0 in cache_subdir_names
