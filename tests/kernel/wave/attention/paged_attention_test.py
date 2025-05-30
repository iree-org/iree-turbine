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
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_arange,
    device_full,
    device_randint,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType, GenericDot, MMAOperand
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
    paged_decode_attention_shape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
    param_bool,
)
from ..common.shapes import get_test_shapes
from typing import List, Optional

# Reference paged attention implementation from vLLM and sglang.
# (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE_KV, BLOCK_SIZE, NUM_SEQS, SEQ_LEN)
shapes = [(16, 1, 64, 64, 32, 2, 100)]
shapes += [(16, 1, 64, 64, 32, 2, 3)]  # small SEQ_LEN test
shapes += [(64, 1, 80, 80, 32, 2, 128)]
shapes += [(128, 2, 80, 80, 32, 2, 500)]
shapes += [(128, 2, 512, 512, 32, 32, 500)]
shapes += [(32, 8, 128, 128, 32, 1319, 1018)]

# Test shapes for MHA paged attention
# (NUM_HEADS, HEAD_SIZE, HEAD_SIZE_KV, BLOCK_SIZE, NUM_SEQS, SEQ_LEN)
mha_shapes = [(16, 64, 64, 32, 2, 100)]
mha_shapes += [(16, 64, 64, 32, 2, 3)]  # small SEQ_LEN test
mha_shapes += [(64, 80, 80, 32, 2, 128)]
mha_shapes += [(128, 80, 80, 32, 2, 500)]


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
    _, num_kv_heads, head_size = key_cache.shape

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


def create_inputs(
    num_seqs: int,
    kv_lens: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    head_size_kv: int,
    dtype: torch.dtype,
):
    query = device_randn(num_seqs, num_query_heads, head_size, dtype=dtype)
    key_cache = device_randn(num_seqs * kv_lens, num_kv_heads, head_size, dtype=dtype)

    value_cache = device_randn(
        num_seqs * kv_lens, num_kv_heads, head_size_kv, dtype=dtype
    )
    block_table = device_arange(num_seqs * kv_lens, dtype=torch.int32).reshape(
        num_seqs, kv_lens
    )
    kv_lens_tensor = device_full((num_seqs,), kv_lens, dtype=torch.int32)
    request_indices = device_zeros(num_seqs + 1, dtype=torch.int32)
    request_indices[1 : num_seqs + 1] = torch.cumsum(kv_lens_tensor, dim=0)
    d = kv_lens // 10
    if d > 0:
        request_indices[1:num_seqs] += device_randint(
            -d, d, (num_seqs - 1,), dtype=torch.int32
        )
    return (
        query,
        key_cache,
        value_cache,
        block_table,
        request_indices,
        kv_lens_tensor,
    )


def create_mha_inputs(
    num_seqs: int,
    kv_lens: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    query = device_randn(num_seqs, num_heads, head_size, dtype=dtype)
    key_cache = device_randn(num_seqs, kv_lens, num_heads, head_size, dtype=dtype)
    value_cache = device_randn(num_seqs, kv_lens, num_heads, head_size, dtype=dtype)
    block_table = device_arange(num_seqs, kv_lens, dtype=torch.int32)
    kv_lens_tensor = device_full((num_seqs,), kv_lens, dtype=torch.int32)
    request_indices = device_zeros(num_seqs + 1, dtype=torch.int32)
    request_indices[1 : num_seqs + 1] = torch.cumsum(kv_lens_tensor, dim=0)
    d = kv_lens // 10
    if d > 0:
        request_indices[1:num_seqs] += device_randint(
            -d, d, (num_seqs - 1,), dtype=torch.int32
        )
    return query, key_cache, value_cache, block_table, request_indices, kv_lens_tensor


def load_inputs(directory):
    query = torch.load(os.path.join(directory, "query.pt"))
    key_cache = torch.load(os.path.join(directory, "key_cache.pt"))
    value_cache = torch.load(os.path.join(directory, "value_cache.pt"))
    block_table = torch.load(os.path.join(directory, "block_table.pt"))
    request_indices = torch.load(os.path.join(directory, "request_indices.pt"))
    kv_lens = torch.load(os.path.join(directory, "kv_lens.pt"))
    return query, key_cache, value_cache, block_table, request_indices, kv_lens


# TODO: Investigate errors on MI250.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@pytest.mark.parametrize("num_kv_splits", [8])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    ],
)
@param_bool("use_wave_runtime", "wr")
def testPagedFlashDecoding(
    shape: tuple[int],
    dtype: torch.dtype,
    enable_scheduling: SchedulingType,
    num_kv_splits: int,
    mfma_variant: MMAType,
    use_wave_runtime: bool,
    request,
):
    torch.manual_seed(0)
    shape = paged_decode_attention_shape(
        num_query_heads=shape[0],
        num_kv_heads=shape[1],
        head_size=shape[2],
        head_size_kv=shape[3],
        block_size=shape[4],
        num_seqs=shape[5],
        kv_lens=shape[6],
    )
    assert shape.num_query_heads % shape.num_kv_heads == 0
    scale = shape.head_size**-0.5

    artifact_directory = None
    if not artifact_directory:
        (
            query,
            key_cache,
            value_cache,
            block_table,
            request_indices,
            kv_lens_tensor,
        ) = create_inputs(
            shape.num_seqs,
            shape.kv_lens,
            shape.num_query_heads,
            shape.num_kv_heads,
            shape.head_size,
            shape.head_size_kv,
            dtype,
        )
    else:
        (
            query,
            key_cache,
            value_cache,
            block_table,
            request_indices,
            kv_lens_tensor,
        ) = load_inputs(artifact_directory)
        shape.num_seqs = query.shape[0]
        shape.num_query_heads = query.shape[1]
        shape.head_size = query.shape[2]
        shape.num_kv_heads = key_cache.shape[2]
        shape.head_size_kv = value_cache.shape[3]

    key_cache_4d = key_cache.view(
        shape.num_seqs, -1, shape.num_kv_heads, shape.head_size
    )
    value_cache_4d = value_cache.view(
        shape.num_seqs, -1, shape.num_kv_heads, shape.head_size_kv
    )

    # Run the wave kernel.
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        num_kv_splits,
        input_dtype=dtype,
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    phase_0_output = device_zeros(
        num_kv_splits,
        shape.num_seqs,
        shape.head_size_kv,
        shape.num_query_heads,
        dtype=torch.float32,
    )
    phase_0_output_max = device_zeros(
        num_kv_splits, shape.num_seqs, shape.num_query_heads, dtype=torch.float32
    )
    output = device_zeros(
        shape.num_seqs, shape.num_query_heads, shape.head_size_kv, dtype=torch.float16
    )

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=use_wave_runtime,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    # TODO: Add variant of non-transposed V attention kernel.
    asm_qk = phase_0(
        query,
        key_cache_4d,
        value_cache_4d,
        request_indices,
        torch.flatten(block_table),
        phase_0_output,
        phase_0_output_max,
    )

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        wave_runtime=use_wave_runtime,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    asm_sv = phase_1(phase_0_output, phase_0_output_max, request_indices, output)

    if dump_generated_mlir:
        filename = f"wave_paged_phase_0_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_qk)
        filename = f"wave_paged_phase_1_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_sv)

    if not artifact_directory:
        # Run the reference implementation.
        ref_vllm_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=torch.ones(shape.num_seqs, dtype=torch.int32),
            kv_lens=kv_lens_tensor,
            block_tables=block_table,
            scale=scale,
            causal=False,
            sliding_window=None,
            soft_cap=None,
        )
    else:
        ref_vllm_output = torch.load(os.path.join(artifact_directory, "output.pt"))

    if dtype == torch.bfloat16:
        assert_close(output, ref_vllm_output, rtol=1e-2, atol=1e-2, check_dtype=False)
    else:
        assert_close(output, ref_vllm_output, rtol=1e-3, atol=1e-3)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", mha_shapes)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@pytest.mark.parametrize("num_kv_splits", [8])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (
            # TODO: we need a very specific dot mma shapes for things to work.
            # Need to better handle vector_shapes/elements_per_thread conflicts
            # when we have multiple MMAs in the same kernel.
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),
        ),
    ],
)
def testPagedFlashDecodingMHA(
    shape: tuple[int],
    dtype: torch.dtype,
    enable_scheduling: SchedulingType,
    num_kv_splits: int,
    mfma_variant: MMAType,
    request,
):
    torch.manual_seed(0)
    shape = paged_decode_attention_shape(
        num_query_heads=shape[0],
        num_kv_heads=shape[0],
        head_size=shape[1],
        head_size_kv=shape[2],
        block_size=shape[3],
        num_seqs=shape[4],
        kv_lens=shape[5],
    )
    assert shape.num_query_heads % shape.num_kv_heads == 0
    scale = shape.head_size**-0.5

    artifact_directory = None
    if not artifact_directory:
        (
            query,
            key_cache,
            value_cache,
            block_table,
            request_indices,
            kv_lens_tensor,
        ) = create_inputs(
            shape.num_seqs,
            shape.kv_lens,
            shape.num_query_heads,
            shape.num_kv_heads,
            shape.head_size,
            shape.head_size_kv,
            dtype,
        )
    else:
        (
            query,
            key_cache,
            value_cache,
            block_table,
            request_indices,
            kv_lens_tensor,
        ) = load_inputs(artifact_directory)
        shape.num_seqs = query.shape[0]
        shape.num_query_heads = query.shape[1]
        shape.head_size = query.shape[2]
        shape.num_kv_heads = key_cache.shape[2]
        shape.head_size_kv = value_cache.shape[3]

    key_cache_4d = key_cache.view(
        shape.num_seqs, -1, shape.num_kv_heads, shape.head_size
    )
    value_cache_4d = value_cache.view(
        shape.num_seqs, -1, shape.num_kv_heads, shape.head_size_kv
    )

    # Run the wave kernel.
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        num_kv_splits,
        input_dtype=dtype,
        mha=True,
    )
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    phase_0_output = device_zeros(
        num_kv_splits,
        shape.num_seqs,
        shape.head_size_kv,
        shape.num_query_heads,
        dtype=torch.float32,
    )
    phase_0_output_max = device_zeros(
        num_kv_splits, shape.num_seqs, shape.num_query_heads, dtype=torch.float32
    )
    output = device_zeros(
        shape.num_seqs, shape.num_query_heads, shape.head_size_kv, dtype=torch.float16
    )

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    # TODO: Add variant of non-transposed V attention kernel.
    asm_qk = phase_0(
        query,
        key_cache_4d,
        value_cache_4d,
        request_indices,
        torch.flatten(block_table),
        phase_0_output,
        phase_0_output_max,
    )

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    asm_sv = phase_1(phase_0_output, phase_0_output_max, request_indices, output)

    if dump_generated_mlir:
        filename = f"wave_paged_mha_phase_0_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_qk)
        filename = f"wave_paged_mha_phase_1_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_sv)

    if not artifact_directory:
        # Run the reference implementation.
        ref_vllm_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=torch.ones(shape.num_seqs, dtype=torch.int32),
            kv_lens=kv_lens_tensor,
            block_tables=block_table,
            scale=scale,
            causal=False,
            sliding_window=None,
            soft_cap=None,
        )
    else:
        ref_vllm_output = torch.load(os.path.join(artifact_directory, "output.pt"))

    if dtype == torch.bfloat16:
        assert_close(output, ref_vllm_output, rtol=1e-2, atol=1e-2, check_dtype=False)
    else:
        assert_close(output, ref_vllm_output, rtol=1e-3, atol=1e-3)
