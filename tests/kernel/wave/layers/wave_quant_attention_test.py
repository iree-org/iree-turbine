# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch.testing import assert_close

from iree.turbine.kernel.wave.layers.quant_attention import wave_sdpa_fp8
from ..common.utils import require_e2e, require_cdna3


@require_e2e
@require_cdna3
def test_SDPA_FP8_no_batch():
    # Testing SD layout.
    device = torch.device("cuda:0")
    query = torch.randn([256, 128], device=device)
    key = torch.randn([256, 128], device=device)
    value = torch.randn([256, 128], device=device)
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    wave_output = wave_sdpa_fp8(query, key, value, q_scale, k_scale, v_scale)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query.to(torch.float32) * q_scale,
        key.to(torch.float32) * k_scale,
        value.to(torch.float32) * v_scale,
    )
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna3
def test_SDPA_FP8_1D_batch():
    # Testing BSD layout.
    device = torch.device("cuda:0")
    query = torch.randn([4, 256, 128], device=device)
    key = torch.randn([4, 256, 128], device=device)
    value = torch.randn([4, 256, 128], device=device)
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    wave_output = wave_sdpa_fp8(query, key, value, q_scale, k_scale, v_scale)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query.to(torch.float32) * q_scale,
        key.to(torch.float32) * k_scale,
        value.to(torch.float32) * v_scale,
    )
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna3
def test_SDPA_FP8_3D_batch():
    # Test BHSD layout.
    device = torch.device("cuda:0")
    query = torch.randn([4, 8, 64, 128], device=device)
    key = torch.randn([4, 8, 256, 128], device=device)
    value = torch.randn([4, 8, 256, 128], device=device)
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    wave_output = wave_sdpa_fp8(query, key, value, q_scale, k_scale, v_scale)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query.to(torch.float32) * q_scale,
        key.to(torch.float32) * k_scale,
        value.to(torch.float32) * v_scale,
    )
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)
