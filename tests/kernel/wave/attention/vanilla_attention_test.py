# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.nn import functional as F
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
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
    device_ones,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from ..common.utils import (
    dump_generated_mlir,
    enable_scheduling_barriers,
    expensive_test_param,
    param_bool,
    require_cdna3,
    require_e2e,
)
from ..common.shapes import get_test_shapes
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
    get_bshd_attention_kernel,
    get_bhsd_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.utils.reference_kernel_utils import (
    scaled_dot_product_attention_bhsd,
)


@require_e2e
@pytest.mark.parametrize("input_shape", get_test_shapes("attention"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, expensive_test_param(SchedulingType.MODULO)],
)
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def testTransposedVAttentionPure(
    input_shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shape = AttentionShape(
        num_query_heads=input_shape[0],
        num_kv_heads=input_shape[0],
        query_seq_len=input_shape[1],
        head_size_kv=input_shape[2],
        head_size=input_shape[3],
        kv_seq_len=input_shape[4],
    )
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape, mfma_variant, dynamic_dims, is_v_transposed=True
    )
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
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
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)
    asm = base_attention(q, k, v.permute([0, 2, 1]), output)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, input_shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)


@require_e2e
@pytest.mark.parametrize("input_shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn", [False])
@param_bool("buffer_ops", "buf")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def testAttentionPure(
    input_shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    buffer_ops: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shape = AttentionShape(
        num_query_heads=input_shape[0],
        num_kv_heads=input_shape[0],
        query_seq_len=input_shape[1],
        head_size_kv=input_shape[2],
        head_size=input_shape[3],
        kv_seq_len=input_shape[4],
    )
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(shape, mfma_variant, dynamic_dims)
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
        use_buffer_load_ops=buffer_ops,
        use_buffer_store_ops=buffer_ops,
    )

    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)
    asm = base_attention(q, k, v, output)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, input_shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("all_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@pytest.mark.parametrize("sliding_window", ([-1, 1024]))
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    ],
)
def testAttentionCausal(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    sliding_window: int,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shape = AttentionShape(
        num_query_heads=shape[0],
        num_kv_heads=shape[0],
        query_seq_len=shape[1],
        head_size_kv=shape[2],
        head_size=shape[3],
        kv_seq_len=shape[4],
    )
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=True,
        is_v_transposed=True,
        sliding_window_size=sliding_window,
    )
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
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
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(1)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)
    asm = base_attention(q, k, v.permute([0, 2, 1]), output)
    if sliding_window >= 0:

        def sliding_window_mask(q_seq_length, kv_seq_length, window_size):
            mask = device_ones((q_seq_length, kv_seq_length), dtype=torch.bool)
            mask = mask.tril().triu(-window_size)
            return mask.to(dtype=torch.bool)

        mask = sliding_window_mask(
            shape.query_seq_len, shape.kv_seq_len, sliding_window
        )
        torch_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        torch_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("all_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    ],
)
def testAttentionBSHD(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shape = AttentionShape(
        num_query_heads=shape[0],
        num_kv_heads=shape[0],
        query_seq_len=shape[1],
        head_size_kv=shape[2],
        head_size=shape[3],
        kv_seq_len=shape[4],
    )

    is_causal = False
    is_custom_mask = True
    custom_mask = None

    assert not (
        is_causal and is_custom_mask
    ), "Causal and custom mask cannot be applied together."

    (
        base_attention_func,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )
    q_shape = (1, shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (1, shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (1, shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())
    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
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
    o_shape = (1, shape.query_seq_len, shape.num_query_heads, shape.head_size_kv)
    output = device_zeros(o_shape, dtype=torch.float32)

    if is_custom_mask:
        custom_mask = device_randn([1, shape.query_seq_len], dtype=torch.float32)
        custom_mask = (custom_mask > 0).int()

        asm = base_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            custom_mask.to(torch.int8),
            output,
        )
    else:
        asm = base_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            output,
        )

    # Torch reference needs to be in BHSD format
    torch_ref = scaled_dot_product_attention_bhsd(
        q, k, v, is_causal=is_causal, custom_mask=custom_mask
    )

    assert_close(
        output.transpose(1, 2),
        torch_ref,
        check_dtype=False,
        atol=1e-3,
        rtol=1e-3,
    )


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("bhsd_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
    ],
)
def testAttentionBHSDCausal(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    shape = AttentionShape(
        batch_size=shape[0],
        num_query_heads=shape[1],
        num_kv_heads=shape[1],
        query_seq_len=shape[2],
        head_size_kv=shape[3],
        head_size=shape[4],
        kv_seq_len=shape[5],
    )

    is_causal = True
    is_custom_mask = False
    custom_mask = None

    assert not (
        is_causal and is_custom_mask
    ), "Causal and custom mask cannot be applied together."

    (
        base_attention_func,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_bhsd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims,
        is_causal=is_causal,
        is_custom_mask=is_custom_mask,
    )
    q_shape = (
        shape.batch_size,
        shape.num_query_heads,
        shape.query_seq_len,
        shape.head_size,
    )
    k_shape = (
        shape.batch_size,
        shape.num_query_heads,
        shape.kv_seq_len,
        shape.head_size,
    )
    v_shape = (
        shape.batch_size,
        shape.num_query_heads,
        shape.kv_seq_len,
        shape.head_size_kv,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention_func)

    torch.manual_seed(1)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)

    # This variant of wave kernel is BHSD
    o_shape = (
        shape.batch_size,
        shape.num_query_heads,
        shape.query_seq_len,
        shape.head_size_kv,
    )
    output = device_zeros(o_shape, dtype=torch.float32)

    if is_custom_mask:
        custom_mask = device_randn([1, shape.query_seq_len], dtype=torch.float32)
        custom_mask = (custom_mask > 0).int()

        asm = base_attention(
            q,
            k,
            v,
            custom_mask.to(torch.int8),
            output,
        )
    else:
        asm = base_attention(
            q,
            k,
            v,
            output,
        )

    # Torch reference needs to be in BHSD format
    torch_ref = scaled_dot_product_attention_bhsd(
        q, k, v, is_causal=is_causal, custom_mask=custom_mask
    )

    assert_close(
        output,
        torch_ref,
        check_dtype=False,
        atol=1e-3,
        rtol=1e-3,
    )


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionBias(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention_bias(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[B, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            bias_reg = tkw.read(bias)
            x_j = x_j + bias_reg
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )

    options = set_default_run_config(options)
    base_attention_bias = wave_compile(options, base_attention_bias)

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    bias = device_randn(shape[0], shape[1], shape[4], dtype=torch.float32)
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])
    # TODO: Add scaling of QK as part of kernel.
    # TODO: Add variant of non-transposed V attention kernel.
    asm = base_attention_bias(
        q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), bias * log2e, output
    )
    k_t = k.transpose(-1, -2)
    a = torch.matmul(q, k_t) * dk_sqrt
    a += bias
    a = F.softmax(a, dim=-1)
    torch_ref = torch.matmul(a, v)

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    if "gfx94" in options.target:
        assert_close(output, torch_ref, atol=2e-3, rtol=5e-3, check_dtype=False)
    else:
        # TODO: Determine why the error is higher on gfx90.
        assert_close(output, torch_ref, atol=3e-3, rtol=8e-1, check_dtype=False)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionSoftCap(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )
    softcap_val = 15.0

    @tkw.wave(constraints)
    def base_attention_softcap(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        # setting softcap to random value of 15.0
        log2e = 1.44269504089
        soft_cap = tkl.Register[B, M, K2, tkl.f32](softcap_val * log2e)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            x_j = soft_cap * tkw.tanh(x_j / soft_cap)
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        run_bench=run_bench,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )

    options = set_default_run_config(options)
    base_attention_softcap = wave_compile(options, base_attention_softcap)

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    softcap = 15.0
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])
    # TODO: Add scaling of QK as part of kernel.
    # TODO: Add variant of non-transposed V attention kernel.
    asm = base_attention_softcap(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    k_t = k.transpose(-1, -2)
    a = torch.matmul(q, k_t) * dk_sqrt
    a = softcap_val * torch.tanh(a / softcap_val)
    a = F.softmax(a, dim=-1)
    torch_ref = torch.matmul(a, v)

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    if "gfx94" in options.target:
        assert_close(output, torch_ref, atol=2e-3, rtol=5e-3, check_dtype=False)
    else:
        # TODO: Determine why the error is higher on gfx90.
        assert_close(output, torch_ref, atol=3e-3, rtol=8e-1, check_dtype=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize(
    "enable_scheduling",
    [SchedulingType.NONE, expensive_test_param(SchedulingType.MODULO)],
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testAttentionF8(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]
    if mfma_variant[0] == MMAType.F32_16x16x32_F8:
        Mvec = 16
        Nvec = 16
    if mfma_variant[0] == MMAType.F32_32x32x16_F8:
        Mvec = 32
        Nvec = 32
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the iterate
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            k_reg = tkw.read(k)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f8 = tkw.cast(e_delta, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f8, new_acc, mfma_variant[1])
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=run_bench,
        dynamic_symbols=[],
        dynamic_symbols_map={},
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )

    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])
    # TODO: Add scaling of QK as part of kernel.
    # TODO: Add variant of non-transposed V attention kernel.
    asm = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )
    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)
    rmse = torch.sqrt(torch.mean(torch.square(output - torch_ref)))
    assert rmse <= 0.006
