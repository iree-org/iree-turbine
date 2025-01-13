# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import math
import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
    device_arange,
    device_randn,
    device_randint,
    device_randn_like,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
)
import os
from torch.testing import assert_allclose
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes
from typing import List, Optional

# Reference paged attention implementation from vLLM and sglang.
# From: https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flash_attn.py
NUM_HEADS = [(128, 4)]
HEAD_SIZES = [64]
BLOCK_SIZES = [64]
DTYPES = [torch.float16]
NUM_BLOCKS = [128]
# First item is query length, second item is key/value length.
# In decode, query length is always one.
# TODO: Check with more queries and unaligned shapes.
SEQ_LENS = [[(1, 16), (1, 8)]]


# From: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/torch_native_backend.py
def _run_sdpa_forward_decode(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    """Run the decode forward by using torch native sdpa op.

    Args:
        query: [num_tokens, num_heads, head_size]
        output: [num_tokens, num_heads, head_size]
        k_cache: [max_total_num_tokens, num_heads, head_size]
        v_cache: [max_total_num_tokens, num_heads, head_size]
        req_to_token: [max_num_reqs, max_context_len]
        req_pool_indices: [num_seqs]
        seq_lens: [num_seqs]
        scaling: float or None
        enable_gqa: bool
        causal: bool

    Returns:
        output: [num_tokens, num_heads, head_size]
    """

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        # TODO: this loop process a sequence per iter, this is inefficient.
        # Need optimize the performance later.

        seq_len_q = 1
        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        per_req_out = (
            torch.nn.functional.scaled_dot_product_attention(
                per_req_query.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q, :, :] = per_req_out
        start_q, start_kv = end_q, end_kv

    return output


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    causal: Optional[bool] = False,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        block_indices = block_tables[i, :kv_len]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if causal:
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@require_e2e
@pytest.mark.parametrize("seq_lens", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    ],
)
def testPagedFlashDecoding(
    seq_lens: List[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    soft_cap: Optional[float],
    num_blocks: int,
    enable_scheduling: bool,
    mfma_variant: MMAType,
    request,
):

    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = device_randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = device_randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = device_randn_like(key_cache)
    # TODO: The block table entries should be able to be a random number
    # in the range [0, num_blocks * block_size), but that fails for now.
    # As a workaround, the maximum value is set to num_seqs - 1.
    block_table = device_randint(0, num_seqs, (num_seqs, max_kv_len), dtype=torch.int32)
    request_indices = device_arange(num_seqs, dtype=torch.int32)
    kv_lens_tensor = device_zeros(num_seqs, dtype=torch.int32)
    for i in range(len(kv_lens)):
        kv_lens_tensor[i] = kv_lens[i]

    # Run the wave kernel.
    # TODO: Currently all but K1 is set to dynamic. This may not be the case.
    S = num_seqs
    B = num_query_heads
    K1 = head_size
    K2 = block_size
    M = 1
    N = head_size
    BH = num_kv_heads
    shape = (B, M, N, K1, K2, BH, S)
    num_kv_splits = 8
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_map_0,
        dynamic_symbols_1,
        dynamic_symbols_map_1,
    ) = get_paged_decode_attention_kernels(
        shape, num_blocks * block_size, mfma_variant, num_kv_splits
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())
    config = get_default_run_config()
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

    sym_U = index_symbol("U")
    if sym_U in hyperparams_0:
        U = hyperparams_0[sym_U]
    else:
        U = dynamic_symbols_map_0[sym_U]
    phase_0_output = device_zeros(U, S, N, B, dtype=torch.float32)
    phase_0_output_max = device_zeros(U, S, B, dtype=torch.float32)
    output = device_zeros(S, B, N, dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / K1)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols_0,
        dynamic_symbols_map=dynamic_symbols_map_0,
    ):
        # TODO: Add scaling of QK as part of kernel.
        mb_qk = phase_0(
            query * dk_sqrt * log2e,
            key_cache.permute([0, 2, 1, 3]),
            value_cache.permute([0, 2, 3, 1]),
            request_indices,
            kv_lens_tensor,
            block_table,
            phase_0_output,
            phase_0_output_max,
        )

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols_1,
        dynamic_symbols_map=dynamic_symbols_map_1,
    ):
        # TODO: Add variant of non-transposed V attention kernel.
        mb_sv = phase_1(phase_0_output, phase_0_output_max, output)

    if dump_generated_mlir:
        filename = f"wave_paged_phase_0_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())
        filename = f"wave_paged_phase_1_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_sv.module_op.get_asm())

    # Run the reference implementation (vllm or sglang).
    ref_vllm_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_table,
        scale=scale,
        causal=False,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    compare_sglang = False
    if compare_sglang:
        # Since the query gets scaled in the first call, we don't
        # scale it again below.
        ref_sglang_output = _run_sdpa_forward_decode(
            query=query,
            output=torch.zeros_like(query),
            k_cache=key_cache,
            v_cache=value_cache,
            req_to_token=block_table,
            req_pool_indices=torch.arange(num_seqs),
            seq_lens=torch.tensor(kv_lens, dtype=torch.int32),
            scaling=1,
            enable_gqa=True,
            causal=False,
        )

        assert_allclose(ref_vllm_output, ref_sglang_output, rtol=1e-3, atol=1e-3)

    assert_allclose(output, ref_vllm_output, rtol=1e-3, atol=1e-3)
