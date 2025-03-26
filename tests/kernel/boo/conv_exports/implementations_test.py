# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import torch
from iree.turbine.kernel.boo.conv_exports.conv import Mode, ConvSignature

# Note: Singleton parameters are intentionally included for ease of adding additional configurations to test
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("N", [3])
@pytest.mark.parametrize("C", [2])
@pytest.mark.parametrize("F", [4])
@pytest.mark.parametrize("H", [32])
@pytest.mark.parametrize("W", [16])
@pytest.mark.parametrize("KH", [2, 3])
@pytest.mark.parametrize("KW", [1])
@pytest.mark.parametrize("layout", ["NHWC", "WCHN"])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("groups", [1])  # TODO: test with groups!=1 when implemented
def test_conv_impl(
    dtype, N, C, F, H, W, KH, KW, layout, dilation, padding, stride, groups
):
    to_shape = lambda layout, map: [map[item] for item in layout]
    in_map = {"N": N, "C": C, "H": H, "W": W}
    ker_map = {"N": F, "C": C // groups, "H": KH, "W": KW}
    kwargs = {
        "input_shape": to_shape(layout, in_map),
        "kernel_shape": to_shape(layout, ker_map),
        "dtype": dtype,
        "input_layout": layout,
        "kernel_layout": layout,
        "output_layout": layout,
        "stride": 2 * [stride],
        "padding": 2 * [padding],
        "dilation": 2 * [dilation],
        "mode": Mode.FORWARD,
        "groups": groups,
    }
    fwd_sig = ConvSignature(**kwargs)
    x, w = fwd_sig.get_sample_conv_args(seed=1)
    x = x.to(device="cpu")
    w = w.to(device="cpu")
    x.requires_grad_(True)
    w.requires_grad_(True)
    fwd = fwd_sig.get_nn_module().to(device="cpu")
    kwargs["mode"] = Mode.INPUT_BACKWARD
    bwd_sig = ConvSignature(**kwargs)
    bwd = bwd_sig.get_nn_module().to(device="cpu")
    kwargs["mode"] = Mode.WEIGHT_BACKWARD
    wrw_sig = ConvSignature(**kwargs)
    wrw = wrw_sig.get_nn_module().to(device="cpu")
    y = fwd(x, w)
    y.retain_grad()
    s = y.sum()
    s.backward(retain_graph=True)
    dsdy = y.grad
    dsdx = bwd(dsdy, w)
    dsdw = wrw(dsdy, x)
    rtol = 1e-4
    atol = 1e-4
    bwd_match = torch.allclose(dsdx, x.grad, rtol=rtol, atol=atol)
    wrw_match = torch.allclose(dsdw, w.grad, rtol=rtol, atol=atol)
    if bwd_match and wrw_match:
        return
    if not bwd_match:
        print(f"{dsdx=}")
        print(f"{x.grad=}")
    if not wrw_match:
        print(f"{dsdw=}")
        print(f"{w.grad=}")
    raise RuntimeError(f"{bwd_match=}; {wrw_match=};")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_conv_custom_impl(dtype):
    sig = ConvSignature(
        input_shape=[1, 16, 16, 2],
        kernel_shape=[4, 3, 3, 2],
        shared_layout="NHWC",
        dtype=dtype,
    )
    default_mod = sig.get_nn_module()
    custom_mod = sig.get_nn_module(use_custom=True)
    args = sig.get_sample_conv_args(seed=10)
    y_ref = default_mod(*args)
    y = custom_mod(*args)
    assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
