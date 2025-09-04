# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Testing functionality of the default BOO backend.
"""
import pytest
import torch
from torch import fx
from torch._dynamo.testing import EagerAndRecordGraphs

from iree.turbine.dynamo.backends import boo


def test_custom_boo_conv_used():
    """Test that we're using our custom convolution op"""
    recorder = EagerAndRecordGraphs()
    compiled_conv = torch.compile(
        torch.ops.aten.convolution,
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
    [call_node] = [n for n in compiled_module.graph.nodes if n.op == "call_function"]
    # Make sure we're using 'boo.ops.convolution_replacement'. We have to do a
    # string check unfortunately, as the target is a fused custom op that we
    # can't inspect.
    call_node_target_str = str(call_node.target)
    assert call_node_target_str.startswith("boo.fused_op_convolution_replacement_")


def test_filter_transpose_conv():
    """Test that we don't offload transpose conv to IREE/BOO."""
    recorder = EagerAndRecordGraphs()
    compiled_conv = torch.compile(
        torch.ops.aten.convolution,
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
    [call_node] = [n for n in compiled_module.graph.nodes if n.op == "call_function"]
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
    with torch.device(device):
        conv = torch.ops.aten.convolution
        boo_conv = torch.compile(conv, backend="iree_boo")
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


@pytest.mark.xfail(
    condition=not torch.cuda.is_available(),
    reason="CPU compile failure potentially related to single dispatch.",
)
def test_boo_layer_norm_used():
    """Test that we're using BOO custom layer norm op."""

    # Force everything to be on a GPU since this fails to compile on CPU.
    device = torch.device("cuda")
    input = torch.randn((10, 20, 30), device=device)
    normalized_shape = (30,)

    recorder = EagerAndRecordGraphs()
    compiled_layer_norm = torch.compile(
        torch.nn.LayerNorm(
            normalized_shape, elementwise_affine=False, bias=False, device=device
        ),
        backend=boo.backend(nested_backend=recorder),
    )

    compiled_layer_norm(input)

    [compiled_module] = recorder.graphs
    assert isinstance(compiled_module, fx.GraphModule)
    print(compiled_module)
    node_target: str | None = None
    for node in compiled_module.graph.nodes:
        if node.op == "call_function" and "boo." in str(node.target):
            node_target = node.target
            break

    # Make sure we are using a BOO op.
    assert node_target is not None, "No BOO op found in the graph"
    assert "fused_op_native_layer_norm" in str(node_target)
