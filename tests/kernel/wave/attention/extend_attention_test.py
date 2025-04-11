# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
import torch
from torch.nn import functional as F
from dataclasses import replace

import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from iree.turbine.kernel.wave.templates.extend_attention_rpe import (
    get_extend_attention_rpe_kernel,
)
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import os
from enum import Enum
from torch.testing import assert_close

from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
    param_bool,
)
from ..common.shapes import get_test_shapes, construct_test_name
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask


def t5_rpe_masked_cond(
    rpe: torch.Tensor, max_rpe_context_length: int, sequence_length: int
) -> torch.Tensor:
    positions = torch.arange(sequence_length).to(device=rpe.device)
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = ((pos_diff >= 0) & (pos_diff < max_rpe_context_length)).to(device=rpe.device)
    rpe_cond = device_zeros(sequence_length, sequence_length, dtype=rpe.dtype)
    rpe_cond[mask] = rpe[pos_diff[mask]]
    return rpe_cond


class ScoreMod(Enum):
    SoftCap = 0
    RPE = 1


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_extend: int,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
):

    cu_seq_lens = [0] * (len(b_seq_len) + 1)
    for i, seq_len in enumerate(b_seq_len):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(b_seq_len)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        qkv_len = end - start
        Q = q[start:end].permute(1, 0, 2)
        K = k[start:end].permute(1, 0, 2)
        K = K.expand(Q.shape[0], *K.shape[1:])
        V = v[start:end].permute(1, 0, 2)
        V = V.expand(Q.shape[0], *V.shape[1:])
        dk_sqrt = math.sqrt(1.0 / Q.shape[-1])
        a = torch.bmm(Q * dk_sqrt, K.transpose(-1, -2))
        if score_mod == ScoreMod.SoftCap:
            a = a / logit_cap
            a = torch.tanh(a)
            a = a * logit_cap
        else:
            rpe_cond = t5_rpe_masked_cond(
                rpe_bias,
                max_rpe_context_length=max_rpe_context_length,
                sequence_length=K.shape[1],
            )
            rpe_cond = rpe_cond.unsqueeze(0)
            rpe_cond = rpe_cond.expand(Q.shape[0], *rpe_cond.shape[1:])
            a = a + rpe_cond
        if is_causal:
            # Create a mask for the upper triangular part (excluding the diagonal)
            mask = (
                torch.triu(torch.ones(a.shape[-2:]), diagonal=1)
                .unsqueeze(0)
                .expand(a.shape)
            )
            # Apply the mask to set the upper triangular part to -infinity
            a[mask == 1] = float("-inf")
        reference = torch.bmm(F.softmax(a, dim=-1).to(dtype=V.dtype), V)
        reference = reference.squeeze(0).permute(1, 0, 2)
        o[start:end] = reference

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
    max_len_extend: int,
    extend_token_num: int,
    dtype: torch.dtype,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
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
        max_len_extend,
        is_causal,
        logit_cap=logit_cap,
        rpe_bias=rpe_bias,
        score_mod=score_mod,
        max_rpe_context_length=max_rpe_context_length,
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
    torch.manual_seed(0)
    b_seq_len_prefix = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_prefix:
        b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
    b_seq_len_extend = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_extend:
        b_seq_len_extend.fill_(shape.fixed_seq_len_extend)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_req_idx = device_arange(B, dtype=torch.int32)
    b_start_loc = device_zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = device_zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = device_zeros((B + 1,), dtype=torch.int32)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = device_zeros((b_seq_len_prefix.sum().item(),), dtype=torch.int32)

    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
        )
    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )
    v_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )

    k_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    q_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)
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
        q_extend[extend_start:extend_end] = device_empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype
        ).normal_(mean=0.1, std=0.2)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = device_zeros((B + 1,), dtype=torch.int32)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)
    logit_cap = 30.0

    b_seq_mask_len = b_seq_len_extend * b_seq_len
    # NOTE: Custom mask is of causal nature in this test. Random mask numerics
    # is not tested.
    custom_mask = device_full(
        (b_seq_mask_len.sum().item(),), fill_value=1, dtype=torch.int8
    )
    mask_offsets = device_zeros((B + 1,), dtype=torch.int32)
    mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
    for i in range(B):
        causal_mask = (
            torch.tril(
                device_full(
                    (b_seq_len_extend[i], b_seq_len_extend[i]),
                    fill_value=1,
                    dtype=torch.int8,
                ),
                diagonal=0,
            )
            == 1
        )
        prefix_mask = device_full(
            (b_seq_len_extend[i], b_seq_len_prefix[i]), fill_value=1, dtype=torch.int8
        )
        mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
        custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

    max_rpe_context_length = 10
    rpe_bias = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
    rpe_bias.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
    rpe_bias[max_rpe_context_length] = 0

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        b_req_idx,
        b_seq_len,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        b_start_loc,
        b_seq_len_prefix,
        extend_token_num,
        max_len_extend,
        logit_cap,
        rpe_bias,
        max_rpe_context_length,
    )


# TODO: Investigate errors on MI250.
# TODO: See why wave_runtime is failing on OSSCI.
# TODO: Push up a setup.py change to make WRT more stable.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("extend"))
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("is_causal", "causal")
@param_bool("use_buffer_ops", "buf_ops")
@param_bool("use_wave_runtime", "wr", [False])
@param_bool("use_custom_mask", "cmask")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def testExtendAttention(
    shape: AttentionShape,
    dtype: torch.dtype,
    enable_scheduling: SchedulingType,
    is_causal: bool,
    use_buffer_ops: bool,
    use_wave_runtime: bool,
    use_custom_mask: bool,
    mfma_variant: MMAType,
    request,
):
    if is_causal and use_custom_mask:
        pytest.skip(
            "Skipping test because causal and custom mask cannot be True simultaneously"
        )
    torch.manual_seed(0)
    assert shape.num_query_heads % shape.num_kv_heads == 0
    (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        b_req_idx,
        b_seq_len,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        b_start_loc,
        b_seq_len_prefix,
        extend_token_num,
        max_len_extend,
        logit_cap,
        _,
        _,
    ) = create_inputs(shape, dtype)
    shape = replace(shape, max_seq_len=max_len_extend)
    shape = replace(shape, flattened_mask_len=custom_mask.shape[0])
    if mfma_variant[0] == MMAType.F32_16x16x16_F16:
        num_waves = 4
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        num_waves = 2

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
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        is_causal=is_causal,
        logit_cap=logit_cap,
        num_waves=num_waves,
        use_custom_mask=use_custom_mask,
    )
    hyperparams.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    perf_filename = construct_test_name(
        "wave_extend_attention", mfma_variant, is_causal, shape
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
        benchmark_batch_size=1000,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, perf_filename) if dump_perf else None
        ),
        dump_intermediates="./inter",
        gpu_native_math_precision=True,
        wave_runtime=(True if use_wave_runtime else False),
    )
    options = set_default_run_config(options)
    extend_attention = wave_compile(options, extend_attention)

    if use_custom_mask:
        asm_qk = extend_attention(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            output,
        )
    else:
        asm_qk = extend_attention(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            output,
        )

    if dump_generated_mlir:
        filename = f"wave_extend_attention_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_qk)

    # Run the reference implementation.
    ref_output = ref_extend_attn(
        q_extend=q_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        b_req_idx=b_req_idx,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        b_seq_len_prefix=b_seq_len_prefix,
        max_len_extend=max_len_extend,
        extend_token_num=extend_token_num,
        dtype=dtype,
        is_causal=(
            is_causal or use_custom_mask
        ),  # Custom mask is set to a causal mask.
        logit_cap=logit_cap,
    )

    assert_close(output, ref_output, rtol=1e-3, atol=1e-3, check_dtype=False)


# TODO: Investigate errors on MI250.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("extend"))
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("is_causal", "causal", [True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def testExtendRpeAttention(
    shape: AttentionShape,
    dtype: torch.dtype,
    enable_scheduling: SchedulingType,
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
        b_req_idx,
        b_seq_len,
        qo_indptr,
        kv_indptr,
        kv_indices,
        _,
        _,
        b_start_loc,
        b_seq_len_prefix,
        extend_token_num,
        max_len_extend,
        logit_cap,
        rpe_bias,
        max_rpe_context_length,
    ) = create_inputs(shape, dtype)
    shape = replace(shape, max_seq_len=max_len_extend)

    if mfma_variant[0] == MMAType.F32_16x16x16_F16:
        num_waves = 4
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        num_waves = 2

    # Run the wave kernel.
    output = device_zeros(
        extend_token_num, shape.num_query_heads, shape.head_size, dtype=torch.float32
    )
    (
        extend_attention_rpe,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_rpe_kernel(
        shape,
        mfma_variant,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        is_causal=is_causal,
        num_waves=num_waves,
        max_rpe_context_length=max_rpe_context_length,
    )
    hyperparams.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    perf_filename = construct_test_name(
        "wave_extend_attention", mfma_variant, is_causal, shape
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        benchmark_batch_size=1000,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, perf_filename) if dump_perf else None
        ),
    )
    options = set_default_run_config(options)
    extend_attention_rpe = wave_compile(options, extend_attention_rpe)

    asm_qk = extend_attention_rpe(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        rpe_bias,
        output,
    )

    if dump_generated_mlir:
        filename = f"wave_extend_attention_kernel_rpe_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_qk)

    # Run the reference implementation.
    ref_output = ref_extend_attn(
        q_extend=q_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        b_req_idx=b_req_idx,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        b_seq_len_prefix=b_seq_len_prefix,
        max_len_extend=max_len_extend,
        extend_token_num=extend_token_num,
        dtype=dtype,
        is_causal=is_causal,
        rpe_bias=rpe_bias,
        logit_cap=logit_cap,
        score_mod=ScoreMod.RPE,
        max_rpe_context_length=max_rpe_context_length,
    )

    torch.testing.assert_close(
        output, ref_output, rtol=5e-2, atol=5e-2, check_dtype=False
    )
