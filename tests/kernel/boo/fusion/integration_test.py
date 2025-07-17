# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Testing functionality of the default BOO backend.
"""
import pytest
import tempfile

from pathlib import Path

import torch

from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs

from iree.turbine.dynamo.backends import boo
from iree.turbine.kernel.boo.runtime import set_cache_dir


def test_custom_boo_conv_used():
    """Test that we're using our custom convolution op"""
    with tempfile.TemporaryDirectory() as td:
        set_cache_dir(Path(td))
        recorder = EagerAndRecordGraphs()
        compiled_conv = torch.compile(
            torch.ops.aten.convolution,
            dynamic=False,
            backend=boo.backend(nested_backend=recorder),
        )

        N, C, H, W = (1, 16, 4, 4)
        F = 32
        K = 3
        input = torch.randn((N, C, H, W))
        weight = torch.randn((F, C, K, K))
        compiled_conv(
            input,
            weight,
            None,  # bias
            [1, 1],  # stride
            [1, 1],  # padding
            [1, 1],  # dilation
            False,  # transposed
            [10, 10],  # output_padding
            1,  # groups
        )

        [compiled_module] = recorder.graphs
        assert isinstance(compiled_module, fx.GraphModule)
        [call_node] = [
            n for n in compiled_module.graph.nodes if n.op == "call_function"
        ]
        # Make sure we're using 'boo.ops.convolution_replacement'. We have to do a
        # string check unfortunately, as the target is a fused custom op that we
        # can't inspect.
        call_node_target_str = str(call_node.target)
        assert call_node_target_str.startswith("boo.fused_op_convolution_replacement_")


def test_filter_transpose_conv():
    """Test that we're using our custom convolution op"""
    with tempfile.TemporaryDirectory() as td:
        set_cache_dir(Path(td))
        recorder = EagerAndRecordGraphs()
        compiled_conv = torch.compile(
            torch.ops.aten.convolution,
            dynamic=False,
            backend=boo.backend(nested_backend=recorder),
        )

        N, C, H, W = (1, 16, 4, 4)
        F = 32
        K = 3
        output_shape = (N, F, H - K + 1, W - K + 1)
        grad_output = torch.randn(output_shape)
        weight = torch.randn((F, C, K, K))
        compiled_conv(
            grad_output,
            weight,
            None,  # bias
            [1, 1],  # stride
            [1, 1],  # padding
            [1, 1],  # dilation
            True,  # transposed
            [0, 0],  # output_padding
            1,  # groups
        )

        [compiled_module] = recorder.graphs
        assert isinstance(compiled_module, fx.GraphModule)
        [call_node] = [
            n for n in compiled_module.graph.nodes if n.op == "call_function"
        ]
        # Make sure we didn't replace the aten convolution.
        assert call_node.target == torch.ops.aten.convolution.default


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires GPU"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "memory_format", [torch.contiguous_format, torch.channels_last]
)
def test_output_layout(device: str, memory_format: torch.memory_format):
    """Test that we properly match the layout of pytorch's implementation."""
    with torch.device(device) and tempfile.TemporaryDirectory() as td:
        set_cache_dir(Path(td))
        conv = torch.ops.aten.convolution
        boo_conv = torch.compile(conv, dynamic=False, backend="iree_boo")
        N, C, H, W = (1, 16, 4, 4)
        F = 32
        K = 3
        input = torch.randn((N, C, H, W)).to(memory_format=memory_format)
        weight = torch.randn((F, C, K, K)).to(memory_format=memory_format)
        args = (
            input,
            weight,
            None,  # bias
            [1, 1],  # stride
            [1, 1],  # padding
            [1, 1],  # dilation
            False,  # transposed
            [10, 10],  # output_padding
            1,  # groups
        )

        actual_result = boo_conv(*args)
        expected_result = conv(*args)
        assert isinstance(actual_result, torch.Tensor)
        assert isinstance(expected_result, torch.Tensor)
        assert actual_result.shape == expected_result.shape
        assert actual_result.stride() == expected_result.stride()
