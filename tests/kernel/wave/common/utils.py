# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
from iree.turbine.kernel.wave.utils import (
    get_default_arch,
)
import torch
import torch.nn.functional as F
from torch import Tensor

require_e2e = pytest.mark.require_e2e
require_cdna2 = pytest.mark.skipif(
    "gfx90" not in get_default_arch(),
    reason="Default architecture is not CDNA2, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(),
    reason="Default architecture is not CDNA3, default architecture is '{}'".format(
        get_default_arch()
    ),
)
# Whether to dump the generated MLIR module.
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)


def param_bool(name, shortname=None, values=None):
    shortname = shortname or name
    values = values or [False, True]
    ids = [f"{shortname}" if v else f"no_{shortname}" for v in values]
    return pytest.mark.parametrize(name, [pytest.param(v) for v in values], ids=ids)


def scaled_dot_product_attention_bhsd(
    query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
) -> Tensor:
    """
    This version mimics PyTorch's `torch.nn.functional.scaled_dot_product_attention`
    with optional causal masking and improved numerical stability.
    Intended for comparison and debugging purposes.

    Args:
        query (Tensor): query tensor of shape [B, H, S_q, D].
        key (Tensor): key tensor of shape [B, H, S_k, D].
        value (Tensor): value tensor of shape [B, H, S_k, D].
        is_causal (bool): If True, applies causal masking to the attention logits.

    Returns:
        Tensor: output tensor of shape [B, H, S_q, D] after applying attention.
    """
    scale: float = query.shape[-1] ** -0.5
    attn_logits: Tensor = torch.matmul(query, key.transpose(-2, -1)) * scale

    if is_causal:
        seq_len_q, seq_len_k = attn_logits.shape[-2], attn_logits.shape[-1]
        causal_mask: Tensor = torch.tril(
            torch.ones(
                (seq_len_q, seq_len_k), device=attn_logits.device, dtype=torch.bool
            )
        )
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))

    # Improve numerical stability using log-sum-exp trick
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
    attn_weights: Tensor = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, value)
