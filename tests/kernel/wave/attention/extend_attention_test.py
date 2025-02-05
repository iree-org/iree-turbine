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
    device_randint,
    device_zeros,
    device_empty,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
import os
from torch.testing import assert_allclose
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes

# Reference paged attention implementation from vLLM and sglang.
shapes = [
    AttentionShape(
        num_seqs=2,
        context_len=1024,
        num_query_heads=16,
        num_kv_heads=1,
        head_size=128,
        head_size_kv=128,
        block_size=64,
    )
]


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_in_batch: int,
    is_causal: bool = False,
):
    cu_seq_lens = [0] * (len(b_seq_len) + 1)
    for i, seq_len in enumerate(b_seq_len):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(b_seq_len)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        o_torch = torch.nn.functional.scaled_dot_product_attention(
            q[start:end].permute(1, 0, 2),
            k[start:end].permute(1, 0, 2),
            v[start:end].permute(1, 0, 2),
            is_causal=is_causal,
            enable_gqa=True,
        ).permute(1, 0, 2)
        o[start:end] = o_torch

    return o


# From: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L369
def ref_extend_attn(
    q_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_seq_len_prefix: torch.Tensor,
    max_len_in_batch: int,
    extend_token_num: int,
    dtype: torch.dtype,
    is_causal: bool = False,
) -> torch.Tensor:
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = device_empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )
    o_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer,
        k_buffer,
        v_buffer,
        o_buffer,
        b_start_loc,
        b_seq_len,
        max_len_in_batch,
        is_causal,
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend

    return o_extend


def create_inputs(
    shape: AttentionShape,
    dtype: torch.dtype,
):

    dtype = torch.float16
    N_CTX = shape.context_len
    B = shape.num_seqs
    H_KV = shape.num_kv_heads
    H_Q = shape.num_query_heads
    D = shape.head_size
    # TODO: Enable when we have proper masking for attention.
    # b_seq_len_prefix = torch.randint(
    #     1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
    # )
    b_seq_len_prefix = torch.empty((B,), dtype=torch.int32, device="cuda")
    for i in range(B):
        b_seq_len_prefix[i] = shape.block_size * (i + 1)
    # TODO: Enable when we have proper masking for attention.
    # b_seq_len_extend = torch.randint(
    #     1, N_CTX // 2, (B,), dtype=torch.int32, device="cuda"
    # )
    b_seq_len_extend = torch.empty((B,), dtype=torch.int32, device="cuda")
    for i in range(B):
        b_seq_len_extend[i] = shape.block_size * (i + 2)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend
    max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

    b_req_idx = torch.arange(B, dtype=torch.int32, device="cuda")
    req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device="cuda")
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    for i in range(B):
        req_to_tokens[i, : b_seq_len[i]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len[i]
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device="cuda"
    ).normal_(mean=0.1, std=0.2)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device="cuda")
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device="cuda")
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = torch.empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype, device="cuda"
        ).normal_(mean=0.1, std=0.2)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_seq_len_extend,
        b_start_loc,
        b_start_loc_extend,
        b_seq_len_prefix,
        max_len_in_batch,
        extend_token_num,
        max_len_extend,
    )


# TODO: Investigate errors on MI250.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [(MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)],
)
def testExtendAttention(
    shape: list[AttentionShape],
    dtype: torch.dtype,
    enable_scheduling: bool,
    is_causal: bool,
    mfma_variant: MMAType,
    request,
):

    torch.manual_seed(0)
    assert shape.num_query_heads % shape.num_kv_heads == 0
    (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        b_seq_len_extend,
        b_start_loc,
        b_start_loc_extend,
        b_seq_len_prefix,
        max_len_in_batch,
        extend_token_num,
        max_len_extend,
    ) = create_inputs(shape, dtype)
    shape.max_seq_len = max_len_extend

    # Run the wave kernel.
    output = device_zeros(
        extend_token_num, shape.num_query_heads, shape.head_size, dtype=torch.float32
    )
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        req_to_tokens.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        is_causal=is_causal,
    )
    hyperparams.update(get_default_scheduling_params())
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

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb_qk = extend_attention(
            q_extend * dk_sqrt * log2e,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            output,
        )

    if dump_generated_mlir:
        filename = f"wave_extend_attention_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())

    # Run the reference implementation.
    ref_output = ref_extend_attn(
        q_extend=q_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        b_req_idx=b_req_idx,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        b_seq_len_prefix=b_seq_len_prefix,
        max_len_in_batch=max_len_in_batch,
        extend_token_num=extend_token_num,
        dtype=dtype,
        is_causal=is_causal,
    )

    assert_allclose(output, ref_output, rtol=1e-3, atol=1e-3)
