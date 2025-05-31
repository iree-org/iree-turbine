# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention_bhsd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    sliding_window: int = -1,
    custom_mask: Tensor | None = None,
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
        Tensor: Output tensor of shape [B, H, S_q, D] after applying attention.
    """
    if query.dtype != torch.float32:
        query = query.to(torch.float32)
    if key.dtype != torch.float32:
        key = key.to(torch.float32)
    if value.dtype != torch.float32:
        value = value.to(torch.float32)

    scale: float = query.shape[-1] ** -0.5
    attn_logits: Tensor = torch.matmul(query, key.transpose(-2, -1)) * scale

    if sliding_window >= 0:
        assert is_causal, f"Sliding window only supported with causal"

    if is_causal:
        seq_len_q, seq_len_k = attn_logits.shape[-2], attn_logits.shape[-1]
        causal_mask: Tensor = torch.tril(
            torch.ones(
                (seq_len_q, seq_len_k), device=attn_logits.device, dtype=torch.bool
            )
        )
        if sliding_window >= 0:
            causal_mask = causal_mask.triu(-sliding_window)
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))

    if custom_mask is not None:
        bool_mask = custom_mask.to(torch.bool)
        bool_mask = bool_mask[:, None, :, None]
        assert bool_mask.shape == (query.shape[0], 1, query.shape[2], 1)
        attn_logits = attn_logits.masked_fill(bool_mask, float("-inf"))

    # Improve numerical stability using log-sum-exp trick
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
    attn_weights: Tensor = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, value)
