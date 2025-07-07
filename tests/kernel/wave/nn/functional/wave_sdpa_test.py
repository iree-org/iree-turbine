# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch.nn import functional as F
from torch.testing import assert_close

import iree.turbine.kernel.wave.nn as wave_nn
from ...common.utils import (
    require_e2e,
    require_cdna3,
)
from iree.turbine.kernel.wave.utils.reference_kernel_utils import (
    scaled_dot_product_attention_bhsd,
)


@require_e2e
@require_cdna3
def test_SDPA_no_batch():
    # Testing SD layout.
    device = torch.device("cuda:0")
    query = torch.randn([256, 128], device=device, dtype=torch.float16)
    key = torch.randn([256, 128], device=device, dtype=torch.float16)
    value = torch.randn([256, 128], device=device, dtype=torch.float16)
    wave_output = wave_nn.functional.wave_sdpa(query, key, value)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
    ).to(torch.float32)
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna3
def test_SDPA_BSD():
    # Testing BSD layout.
    device = torch.device("cuda:0")
    query = torch.randn([4, 256, 128], device=device, dtype=torch.float16)
    key = torch.randn([4, 256, 128], device=device, dtype=torch.float16)
    value = torch.randn([4, 256, 128], device=device, dtype=torch.float16)
    wave_output = wave_nn.functional.wave_sdpa(query, key, value)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
    ).to(torch.float32)
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna3
def test_SDPA_BHSD():
    # Test BHSD layout.
    device = torch.device("cuda:0")
    query = torch.randn([4, 8, 64, 128], device=device, dtype=torch.float16)
    key = torch.randn([4, 8, 256, 128], device=device, dtype=torch.float16)
    value = torch.randn([4, 8, 256, 128], device=device, dtype=torch.float16)
    wave_output = wave_nn.functional.wave_sdpa(query, key, value)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
    ).to(torch.float32)
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)


@require_e2e
@require_cdna3
def test_SDPA_BHSD_causal():
    # Test BHSD layout with causal attention.
    device = torch.device("cuda:0")
    query = torch.randn([4, 8, 64, 128], device=device, dtype=torch.float16)
    key = torch.randn([4, 8, 256, 128], device=device, dtype=torch.float16)
    value = torch.randn([4, 8, 256, 128], device=device, dtype=torch.float16)
    wave_output = wave_nn.functional.wave_sdpa(
        query,
        key,
        value,
        is_causal=True,
    )
    torch_ref = scaled_dot_product_attention_bhsd(
        query,
        key,
        value,
        is_causal=True,
    ).to(torch.float32)
    assert_close(wave_output, torch_ref, atol=1e-3, rtol=1e-3)
