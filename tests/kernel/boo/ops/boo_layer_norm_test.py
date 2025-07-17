# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import torch
import pytest
import tempfile
from pathlib import Path
import math
import unittest

from iree.turbine.kernel.boo.runtime import set_cache_dir, LaunchableRuntimeCache
from iree.turbine.kernel.boo.ops import (
    boo_layer_norm,
    enable_backward,
    disable_backward,
)


# TODO: share this with boo_conv_test.py somehow.
@pytest.fixture
def use_backward():
    """Enables BOO for backward kernels before the test and disables it afterwards."""
    enable_backward()
    yield
    disable_backward()


@pytest.mark.parametrize(
    ("input_grad", "weight_grad", "bias_grad"),
    (
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ),
)
def testBackwardCachePyTorch(input_grad: bool, weight_grad: bool, bias_grad: bool):
    LaunchableRuntimeCache.clear()
    with tempfile.TemporaryDirectory() as td:
        set_cache_dir(Path(td))
        device = "cuda:0" if torch.cuda.is_available() else None
        input = torch.ones(
            [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=input_grad
        )
        weight = torch.ones(
            [16], dtype=torch.float32, device=device, requires_grad=weight_grad
        )
        bias = torch.ones(
            [16], dtype=torch.float32, device=device, requires_grad=bias_grad
        )
        output = boo_layer_norm(input, [16], weight, bias)

        # If none of the gradients are required, backward computation will raise
        # an error. Tell pytest that this is expected.
        context = (
            contextlib.nullcontext()
            if any((input_grad, weight_grad, bias_grad))
            else pytest.raises(RuntimeError)
        )
        with context:
            output.sum().backward()

        items = [x.name for x in Path(td).glob("*/")]
        assert "layer_norm_1d_float32_forward_1x1x16x16_w_b" in items
        # When using Pytorch backward (because we didn't set enabled BOO
        # backward), we shouldn't have kernels generated for these.
        assert "layer_norm_1d_float32_input_backward_1x1x16x16_w_b" not in items
        assert "layer_norm_1d_float32_weight_backward_1x1x16x16_w_b" not in items
        assert "layer_norm_1d_float32_bias_backward_1x1x16x16_w_b" not in items


@pytest.mark.usefixtures("use_backward")
@pytest.mark.parametrize(
    ("input_grad", "weight_grad", "bias_grad"),
    (
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ),
)
def testBackwardCacheBoo(input_grad: bool, weight_grad: bool, bias_grad: bool):
    LaunchableRuntimeCache.clear()
    with tempfile.TemporaryDirectory() as td:
        set_cache_dir(Path(td))
        device = "cuda:0" if torch.cuda.is_available() else None
        input = torch.ones(
            [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=input_grad
        )
        weight = torch.ones(
            [16], dtype=torch.float32, device=device, requires_grad=weight_grad
        )
        bias = torch.ones(
            [16], dtype=torch.float32, device=device, requires_grad=bias_grad
        )
        output = boo_layer_norm(input, [16], weight, bias)

        # If none of the gradients are required, backward computation will raise
        # an error. Tell pytest that this is expected.
        context = (
            contextlib.nullcontext()
            if any((input_grad, weight_grad, bias_grad))
            else pytest.raises(RuntimeError)
        )
        with context:
            output.sum().backward()

        def _validate(name: str, expected: bool):
            if expected:
                assert name in items
            else:
                assert not name in items

        items = [x.name for x in Path(td).glob("*/")]
        assert "layer_norm_1d_float32_forward_1x1x16x16_w_b" in items
        _validate("layer_norm_1d_float32_input_backward_1x1x16x16_w_b", input_grad)
        _validate("layer_norm_1d_float32_weight_backward_1x1x16x16_w_b", weight_grad)
        _validate("layer_norm_1d_float32_bias_backward_1x1x16x16_w_b", bias_grad)


@pytest.mark.usefixtures("use_backward")
class BooLayerNormTest(unittest.TestCase):
    def setUp(self):
        LaunchableRuntimeCache.clear()
        LaunchableRuntimeCache.set_cache_limit(0)

    def testBooLayerNormDefault(self):
        # TODO: consider factoring this out to setUp/tearDown
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            device = "cuda:0" if torch.cuda.is_available() else None
            input = torch.arange(
                0, 2 * 3 * 4 * 5, dtype=torch.float32, device=device
            ).reshape([2, 3, 4, 5])
            weight = torch.ones([5], dtype=torch.float32, device=device) * 2.0
            bias = torch.ones([5], dtype=torch.float32, device=device)
            result = boo_layer_norm(input, [5], weight, bias)

            sq2 = math.sqrt(2.0)
            line = (
                torch.tensor(
                    [-sq2, -sq2 / 2.0, 0.0, sq2 / 2.0, sq2],
                    dtype=torch.float32,
                    device=device,
                )
                * 2.0
                + 1.0
            )
            reference = torch.broadcast_to(line, [2, 3, 4, 5])
            torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)

    def testBooLayerNormBackwards(self):
        with tempfile.TemporaryDirectory() as td:
            set_cache_dir(Path(td))
            device = "cuda:0" if torch.cuda.is_available() else None
            input = torch.arange(
                0, 2 * 3 * 4 * 5, dtype=torch.float32, device=device, requires_grad=True
            ).reshape([2, 3, 4, 5])
            weight = (
                torch.ones([5], dtype=torch.float32, device=device, requires_grad=True)
                * 2.0
            )
            bias = torch.ones(
                [5], dtype=torch.float32, device=device, requires_grad=True
            )
            torch.autograd.gradcheck(
                boo_layer_norm, (input, [5], weight, bias), atol=1e-2, eps=1e-3
            )


if __name__ == "__main__":
    unittest.main()
