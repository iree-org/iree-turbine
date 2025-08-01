# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from pathlib import Path

import torch
import iree.turbine.kernel.boo.ops as boo_ops
import iree.turbine.aot as aot


class LayoutCustomizableSample0(torch.nn.Module):
    def forward(self, x, w):
        return (
            boo_ops.boo_layout_customizable_convolution(
                x, w, None, [1, 1], [0, 0], [1, 1], 1, "NHWC", "NHWC", "NHWC"
            )
            * 0.1
        )


class LayoutCustomizableSample1(torch.nn.Module):
    def forward(self, x, w):
        return (
            boo_ops.convolution_replacement(
                x,
                w,
                None,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )
            * 0.1
        )


devices = [torch.device("cpu")] + (
    [torch.device("cuda:0")] if torch.cuda.is_available() else []
)


@pytest.mark.parametrize("device", devices)
def test_AOT_layout_customizable(device: torch.device):
    N = 2
    C = 32
    H = 16
    W = 16
    k = 1
    f = 2
    x = torch.randn([N, H, W, C], device=device)
    w = torch.randn([f, k, k, C], device=device)
    e = aot.export(LayoutCustomizableSample0(), args=(x, w))
    e.mlir_module.verify()
    assert (
        "call @conv_2d_float32_forward_2x16x16x32_nhwc_2x1x1x32_fhwc_nhwf_1x1s_0x0p_1x1d_1g"
        in str(e.mlir_module)
    )


@pytest.mark.parametrize("device", devices)
def test_AOT_layout_conv_replacement(device: torch.device, boo_cache_dir: Path):
    N = 2
    C = 32
    H = 16
    W = 16
    k = 1
    f = 2
    x = torch.randn([N, C, H, W], device=device).to(memory_format=torch.channels_last)
    w = torch.randn([f, C, k, k], device=device).to(memory_format=torch.channels_last)
    exported_program = torch.export.export(LayoutCustomizableSample1(), args=(x, w))
    gm = exported_program.graph_module
    graph_op = boo_ops.get_custom_graph_op(gm, force_single_dispatch=True)
    y: torch.Tensor = graph_op(x, w)
    assert y.is_contiguous(
        memory_format=torch.channels_last
    ), "Output must be in channels last format."
    cached_items = list(Path.glob(boo_cache_dir, "*/"))
    assert len(cached_items) == 1, f"Expected one cached item, got {cached_items}."
    p = cached_items[0]
    assert p.name.endswith("_2x32x16x16xfloat32_perm_0231_2x32x1x1xfloat32_perm_0231")
