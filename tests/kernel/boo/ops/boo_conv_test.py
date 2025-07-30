# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import pytest

from pathlib import Path

import torch

from iree.turbine.kernel.boo.runtime import LaunchableRuntimeCache, use_cache_dir
from iree.turbine.kernel.boo.ops import (
    boo_conv,
    enable_backward,
    disable_backward,
)


@pytest.fixture
def use_backward():
    enable_backward()
    yield
    disable_backward()


@pytest.mark.parametrize(
    ("x_grad", "w_grad"), ((False, False), (True, False), (False, True), (True, True))
)
def testBackwardCachePytorch(x_grad, w_grad, boo_cache_dir: Path):
    LaunchableRuntimeCache.clear()
    device = "cuda:0" if torch.cuda.is_available() else None
    x = torch.ones(
        [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=x_grad
    )
    w = torch.ones(
        [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=w_grad
    )
    y = boo_conv(x, w, shared_layout="NCHW")

    context = (
        contextlib.nullcontext() if x_grad or w_grad else pytest.raises(RuntimeError)
    )
    with context:
        y.sum().backward()

    items = [x.name for x in boo_cache_dir.glob("*/")]

    assert (
        "conv_2d_float32_forward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g"
        in items
    )
    assert (
        "conv_2d_float32_weight_backward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g"
        not in items
    )
    assert (
        "conv_2d_float32_input_backward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g"
        not in items
    )


def _marked_xfail(*args):
    return pytest.param(
        *args,
        marks=pytest.mark.xfail(
            condition=not torch.cuda.is_available(),
            reason="CPU backward compile failure. Remove when #998 is resolved.",
        ),
    )


@pytest.mark.usefixtures("use_backward")
@pytest.mark.parametrize(
    ("x_grad", "w_grad"),
    (
        (False, False),
        _marked_xfail(True, False),
        (False, True),
        _marked_xfail(True, True),
    ),
)
def testBackwardCacheBoo(x_grad, w_grad, boo_cache_dir: Path):
    LaunchableRuntimeCache.clear()
    device = "cuda:0" if torch.cuda.is_available() else None
    x = torch.ones(
        [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=x_grad
    )
    w = torch.ones(
        [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=w_grad
    )
    y = boo_conv(x, w, shared_layout="NCHW")

    context = (
        contextlib.nullcontext() if x_grad or w_grad else pytest.raises(RuntimeError)
    )
    with context:
        y.sum().backward()

    items = [x.name for x in boo_cache_dir.glob("*/")]

    assert (
        "conv_2d_float32_forward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g"
        in items
    )

    _validate = lambda name, expected: (
        (name in items) if expected else (name not in items)
    )
    assert _validate(
        "conv_2d_float32_weight_backward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
        w_grad,
    )
    assert _validate(
        "conv_2d_float32_input_backward_1x1x16x16_nchw_1x1x2x2_fchw_nfhw_1x1s_0x0p_1x1d_1g",
        x_grad,
    )


@pytest.mark.usefixtures("use_backward")
class TestBooConv:
    @pytest.fixture(autouse=True)
    def setUp(self):
        LaunchableRuntimeCache.clear()
        LaunchableRuntimeCache.set_cache_limit(0)

    def testBooConvNonDefault(self):
        device = "cuda:0" if torch.cuda.is_available() else None
        x = torch.ones([2, 16, 16, 3], dtype=torch.float32, device=device)
        w = torch.ones([4, 2, 2, 3], dtype=torch.float32, device=device)
        y = boo_conv(x, w, shared_layout="NHWC", stride=2, dilation=2)
        y_exp = torch.ones_like(y, device=device) * 12.0
        assert (
            round(torch.abs(y - y_exp).sum().item(), ndigits=7) == 0.0
        ), f"Expected output to be close to splat 12.0 tensor. Got {y}"

    @pytest.mark.xfail(
        condition=not torch.cuda.is_available(),
        reason="CPU backward compile failure. Remove when #998 is resolved.",
    )
    def testBooConvBackwardDefault(self):
        device = "cuda:0" if torch.cuda.is_available() else None
        x = torch.ones(
            [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=True
        )
        w = torch.ones(
            [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=True
        )
        torch.autograd.gradcheck(boo_conv, (x, w), atol=1e-5, eps=1e-3)

    @pytest.mark.xfail(
        condition=not torch.cuda.is_available(),
        reason="CPU backward compile failure. Remove when #998 is resolved.",
    )
    def testBooConvBackwardsWithBias(self):
        device = "cuda:0" if torch.cuda.is_available() else None
        x = torch.ones(
            [1, 1, 16, 16], dtype=torch.float32, device=device, requires_grad=True
        )
        w = torch.ones(
            [1, 1, 2, 2], dtype=torch.float32, device=device, requires_grad=True
        )
        b = torch.ones([1], dtype=torch.float32, device=device, requires_grad=True)
        torch.autograd.gradcheck(boo_conv, (x, w, b), atol=1e-5, eps=1e-3)

    @pytest.mark.xfail(
        reason="CPU backward compile failure. Remove when #998 is resolved."
    )
    def testBooConvBackwardsAmpContextCPU(self, tmp_path: Path):
        """We expect this to not perform autocasting."""

        device = None
        x = torch.ones(
            [1, 1, 32, 32], dtype=torch.float32, device=device, requires_grad=True
        )
        w = torch.ones(
            [1, 1, 4, 4], dtype=torch.float32, device=device, requires_grad=True
        )

        with use_cache_dir(tmp_path / "boo_cache_0") as boo_cache_0:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = boo_conv(x, w, shared_layout="NCHW")
                loss = y.sum()

            loss.backward()

            items = [x.name for x in boo_cache_0.glob("*/")]
            expected_dtype_str = "float32"
            unexpected_dtype_str = "bfloat16"
            assert (
                f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                not in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )

            assert y.dtype == torch.float32
            assert x.grad.dtype == torch.float32
            assert w.grad.dtype == torch.float32
            assert w.dtype == torch.float32

        with use_cache_dir(tmp_path / "boo_cache_1") as boo_cache_1:
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                y = boo_conv(x, w, shared_layout="NCHW")
                loss = y.sum()

            loss.backward()

            items = [x.name for x in boo_cache_1.glob("*/")]
            expected_dtype_str = "bfloat16"
            unexpected_dtype_str = "float32"
            assert (
                f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                not in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )
            assert (
                f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
                in items
            )
            # Make sure we got back the correct original dtypes.
            assert y.dtype == torch.bfloat16
            assert x.grad.dtype == torch.float32
            assert w.grad.dtype == torch.float32
            assert w.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU to test.")
    def testBooConvBackwardsAmpContextCUDA(self, boo_cache_dir: Path):
        device = "cuda:0"
        x = torch.ones(
            [1, 1, 32, 32], dtype=torch.float32, device=device, requires_grad=True
        )
        w = torch.ones(
            [1, 1, 4, 4], dtype=torch.float32, device=device, requires_grad=True
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = boo_conv(x, w, shared_layout="NCHW")
            loss = y.sum()

        loss.backward()
        items = [x.name for x in boo_cache_dir.glob("*/")]
        expected_dtype_str = "bfloat16"
        unexpected_dtype_str = "float32"
        assert (
            f"conv_2d_{unexpected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
            not in items
        )
        assert (
            f"conv_2d_{expected_dtype_str}_forward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
            in items
        )
        assert (
            f"conv_2d_{expected_dtype_str}_weight_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
            in items
        )
        assert (
            f"conv_2d_{expected_dtype_str}_input_backward_1x1x32x32_nchw_1x1x4x4_fchw_nfhw_1x1s_0x0p_1x1d_1g"
            in items
        )
        # Make sure we got back the correct original dtypes.
        assert y.dtype == torch.bfloat16
        assert x.grad.dtype == torch.float32
        assert w.grad.dtype == torch.float32
        assert w.dtype == torch.float32
