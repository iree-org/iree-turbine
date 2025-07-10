# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
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
    quantized_tensor,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from ..common.utils import (
    enable_scheduling_barriers,
    param_bool,
    require_e2e,
    require_cdna3,
)
from ..common.shapes import get_test_shapes
from iree.turbine.kernel.wave.templates.gqa_vanilla_attention import (
    get_gqa_bshd_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.reference_kernel_utils import (
    scaled_dot_product_attention_bhsd,
)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("gqa_bshd_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("causal", "causal")
@pytest.mark.parametrize("sliding_window", [-1, 1024])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    ],
)
def testGQABSHDAttention(
    shape: AttentionShape,
    enable_scheduling: SchedulingType,
    causal: bool,
    sliding_window: int,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    if not causal:
        sliding_window = -1

    (
        base_attention_func,
        hyperparams,
        dynamic_symbols,
    ) = get_gqa_bshd_attention_kernel(
        shape,
        mfma_variant,
        is_causal=causal,
        sliding_window_size=sliding_window,
        input_dtype=torch.float16,
    )
    q_shape = (
        shape.num_seqs,
        shape.num_query_heads,
        shape.query_seq_len,
        shape.head_size,
    )
    k_shape = (shape.num_seqs, shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_seqs, shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())
    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
    )
    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention_func)

    torch.manual_seed(1)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)

    # This variant of wave kernel is BSHD
    o_shape = (
        shape.num_seqs,
        shape.query_seq_len,
        shape.num_query_heads,
        shape.head_size_kv,
    )
    output = device_zeros(o_shape, dtype=torch.float32)

    asm = base_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        output,
    )

    # Torch reference needs to be in BHSD format
    torch_ref = scaled_dot_product_attention_bhsd(
        q, k, v, is_causal=causal, sliding_window=sliding_window
    )

    assert_close(
        output.transpose(1, 2),
        torch_ref,
        check_dtype=False,
        atol=1e-3,
        rtol=1e-3,
    )


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("gqa_bshd_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@pytest.mark.parametrize("sliding_window", [-1, 1024])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testCausalGQABSHDAttentionF8(
    shape: AttentionShape,
    enable_scheduling: SchedulingType,
    sliding_window: int,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    # Sample tensor scaling from Brevitas SDXL-FP8.
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    (
        base_attention_func,
        hyperparams,
        dynamic_symbols,
    ) = get_gqa_bshd_attention_kernel(
        shape,
        mfma_variant,
        is_causal=True,
        sliding_window_size=sliding_window,
        use_fp8=True,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    q_shape = (
        shape.num_seqs,
        shape.num_query_heads,
        shape.query_seq_len,
        shape.head_size,
    )
    k_shape = (shape.num_seqs, shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_seqs, shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())
    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
    )
    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention_func)

    torch.manual_seed(1)

    # Smaller range to help with FP8 minimum range
    MAX_RANGE = 128
    q = quantized_tensor(q_shape, dtype=torch.float16, scale=MAX_RANGE)
    k = quantized_tensor(k_shape, dtype=torch.float16, scale=MAX_RANGE)
    v = quantized_tensor(v_shape, dtype=torch.float16, scale=MAX_RANGE)

    # This variant of wave kernel is BSHD
    o_shape = (
        shape.num_seqs,
        shape.query_seq_len,
        shape.num_query_heads,
        shape.head_size_kv,
    )
    output = device_zeros(o_shape, dtype=torch.float32)

    base_attention(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        output,
    )

    # Torch reference needs to be in BHSD format
    torch_ref = scaled_dot_product_attention_bhsd(
        q.to(torch.float32) * q_scale,
        k.to(torch.float32) * k_scale,
        v.to(torch.float32) * v_scale,
        is_causal=True,
        sliding_window=sliding_window,
    )

    rmse = torch.sqrt(torch.mean(torch.square(output.transpose(1, 2) - torch_ref)))
    # Higher tolerance because, we are testing a higher range
    # of numbers here (-128, 128), typically device_rand just gets (-1, 1).
    assert rmse < 0.04, f"RMSE: {rmse} is too high, expected less than 0.04"
