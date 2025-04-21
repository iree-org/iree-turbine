# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import tempfile
import pytest
import torch
from torch.testing import assert_close
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
import sympy

from iree.turbine.kernel.wave.cache import (
    is_cache_enabled,
    get_cache_manager,
    reset_cache_manager,
)
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
    get_default_arch,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import Constraint, MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from ..common.utils import (
    require_e2e,
)

require_cache = pytest.mark.skipif(
    not is_cache_enabled(), reason="filesystem cache is disabled"
)


def generate_attention_kernel(constraints: list[Constraint]):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Setup transpose mapping
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

        # This microkernel encodes the fact that if the reduction
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
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    return base_attention


@require_e2e
@require_cache
def testSameConfig(tmp_path):
    reset_cache_manager(tmp_path)
    shape = (8, 128, 128, 64, 256)
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
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    mfma_variant = MMAType.F32_32x32x8_F16

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 32, N: 32},
        )
    ]

    base_attention = generate_attention_kernel(constraints)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
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

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    ).to(torch.float32)
    cache_manager = get_cache_manager()

    options = WaveCompileOptions(
        subs=copy.deepcopy(hyperparams),
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)

    # Before compilation, nothing in cache.
    assert (
        len(cache_manager.session_cache) == 0
    ), "Expected to start runtime with no cache."

    base_attention_0 = wave_compile(options, base_attention)

    # First run/call to kernel, this should compile from scratch.
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    mb = base_attention_0(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 0
    ), "Expected first call to not be cached."
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    # Subsequent run/call to kernel, this should be using cached.
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    base_attention_1 = wave_compile(options, base_attention)
    cached_kernel = base_attention_1(
        q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output
    )
    assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected to keep size of cache because we reuse same kernel."
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 1
    ), "Expected subsequent call to be cached."


@require_e2e
@require_cache
def testDifferentDynamicSameBlock(tmp_path):
    reset_cache_manager(tmp_path)
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
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]
    constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    mfma_variant = MMAType.F32_32x32x8_F16

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 32, N: 32},
        )
    ]

    base_attention = generate_attention_kernel(constraints)

    shape_0 = (4, 128, 128, 64, 256)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        K1: shape_0[3],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [M, N, B, K2]
    dynamic_sym_shape0 = {M: shape_0[1], N: shape_0[2], B: shape_0[0], K2: shape_0[4]}

    cache_manager = get_cache_manager()

    # First run/call to kernel, this should compile from scratch.
    options = WaveCompileOptions(
        subs=copy.deepcopy(hyperparams),
        canonicalize=True,
        run_bench=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_sym_shape0,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)

    assert (
        len(cache_manager.session_cache) == 0
    ), "Expected to start runtime with no cache."

    base_attention_0 = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q_shape_0 = device_randn(shape_0[0], shape_0[1], shape_0[3], dtype=torch.float16)
    k_shape_0 = device_randn(shape_0[0], shape_0[4], shape_0[3], dtype=torch.float16)
    v_shape_0 = device_randn(shape_0[0], shape_0[4], shape_0[2], dtype=torch.float16)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape_0[3])
    torch_ref_shape_0 = torch.nn.functional.scaled_dot_product_attention(
        q_shape_0, k_shape_0, v_shape_0, attn_mask=None
    ).to(torch.float32)
    output_shape_0 = device_zeros(
        shape_0[0], shape_0[1], shape_0[2], dtype=torch.float32
    )
    mb = base_attention_0(
        q_shape_0 * dk_sqrt * log2e,
        k_shape_0,
        v_shape_0.permute([0, 2, 1]),
        output_shape_0,
    )
    assert_close(output_shape_0, torch_ref_shape_0, atol=1e-3, rtol=1e-3)
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 0
    ), "Expected first call to not be cached."
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    # Despite having different problem size, since we use exact same
    # block size we should be able to use cache.
    shape_1 = (2, 128, 64, 64, 128)
    dynamic_sym_shape1 = {M: shape_1[1], N: shape_1[2], B: shape_1[0], K2: shape_1[4]}
    options = WaveCompileOptions(
        subs=copy.deepcopy(hyperparams),
        canonicalize=True,
        run_bench=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_sym_shape1,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    base_attention_1 = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q_shape_1 = device_randn(shape_1[0], shape_1[1], shape_1[3], dtype=torch.float16)
    k_shape_1 = device_randn(shape_1[0], shape_1[4], shape_1[3], dtype=torch.float16)
    v_shape_1 = device_randn(shape_1[0], shape_1[4], shape_1[2], dtype=torch.float16)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape_1[3])
    torch_ref_shape_1 = torch.nn.functional.scaled_dot_product_attention(
        q_shape_1, k_shape_1, v_shape_1, attn_mask=None
    ).to(torch.float32)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    # Subsequent run/call to kernel, this should be using cached.
    output_shape_1 = device_zeros(
        shape_1[0], shape_1[1], shape_1[2], dtype=torch.float32
    )
    cached_kernel = base_attention_1(
        q_shape_1 * dk_sqrt * log2e,
        k_shape_1,
        v_shape_1.permute([0, 2, 1]),
        output_shape_1,
    )
    assert_close(output_shape_1, torch_ref_shape_1, atol=1e-3, rtol=1e-3)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected to keep size of cache because we reuse same kernel."
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 1
    ), "Expected subsequent call to be cached."


@require_e2e
@require_cache
def testSameSizeDifferentBlock(tmp_path):
    reset_cache_manager(tmp_path)
    shape = (8, 128, 128, 64, 256)
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
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    mfma_variant = MMAType.F32_32x32x8_F16

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 32, N: 32},
        )
    ]

    base_attention = generate_attention_kernel(constraints)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
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

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])
    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    ).to(torch.float32)
    cache_manager = get_cache_manager()
    options = WaveCompileOptions(
        subs=copy.deepcopy(hyperparams),
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)

    assert (
        len(cache_manager.session_cache) == 0
    ), "Expected to start runtime with no cache."

    base_attention_0 = wave_compile(options, base_attention)

    # First run/call to kernel, this should compile from scratch.
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    mb_config_0 = base_attention_0(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 0
    ), "Expected first call to not be cached."
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    # Subsequent run/call to kernel, this trigger recompile because we use
    # a different block size/config.
    hyperparams[BLOCK_N] = 32
    hyperparams[BLOCK_K2] = 32
    options = WaveCompileOptions(
        subs=copy.deepcopy(hyperparams),
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    base_attention_1 = wave_compile(options, base_attention)

    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    mb_config_1 = base_attention_1(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
    assert (
        len(cache_manager.session_cache) == 2
    ), "Expected cache size to increment, because we use different block size/config."
    assert (
        cache_manager.cache_misses == 2 and cache_manager.cache_hits == 0
    ), "Expected subsequent call to be cached."


# Free vars are variables defined outside the kernels that would impact the
# kernel being generated. For example values of is_causal, sm_scales, and logit_cap.
# This test help ensure WaveCacher do not re-use kernels when free vars are different,
# despite having exact same configurations.


@require_e2e
@require_cache
def testSameConfigDifferentFreeVar(tmp_path):
    reset_cache_manager(tmp_path)
    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
    # Order of shapes: (B, M, N, K1, K2)
    input_shape = (8, 128, 128, 64, 256)
    dynamic_dims = False
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
    cache_manager = get_cache_manager()
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    # TODO: Add scaling of QK as part of kernel.
    # TODO: Add variant of non-transposed V attention kernel.
    non_causal_mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    assert (
        cache_manager.cache_misses == 1 and cache_manager.cache_hits == 0
    ), "Expected first call to not be cached."
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    (
        causal_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape, mfma_variant, dynamic_dims, is_causal=True, is_v_transposed=True
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    causal_attention = wave_compile(options, causal_attention)

    torch.manual_seed(0)
    q = device_randn(q_shape, dtype=torch.float16)
    k = device_randn(k_shape, dtype=torch.float16)
    v = device_randn(v_shape, dtype=torch.float16)
    output = device_zeros(o_shape, dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    # TODO: Add scaling of QK as part of kernel.
    # TODO: Add variant of non-transposed V attention kernel.
    causal_mb = causal_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
    assert (
        cache_manager.cache_misses == 2 and cache_manager.cache_hits == 0
    ), "Expected to be cached despite same config, since it has different values for is_causal."
    assert (
        len(cache_manager.session_cache) == 2
    ), "Expected len == 2, after caching second kernel."


# This test is important to check two things:
# 1. We can cache nested functions
# 2. We do not cache if function signature is different even though
#    core is same.


@require_e2e
@require_cache
def testDifferentSignatureSameCore(tmp_path):
    reset_cache_manager(tmp_path)
    shape = [256, 256]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    def add(a, b):
        return a + b

    def core(a):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        double = add(res, res)
        tkw.write(double, a, elements_per_thread=ELEMS_PER_THREAD)

    @tkw.wave(constraints)
    def double(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        core(a)

    @tkw.wave(constraints)
    def double_transpose(a: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16]):
        core(a)

    cache_manager = get_cache_manager()
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    )
    options = set_default_run_config(options)
    assert (
        len(cache_manager.session_cache) == 0
    ), "Expected len == 0, before any compilation of kernel."

    double0_fn = wave_compile(options, double)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    double1_fn = wave_compile(options, double)
    # This used to break because in nested function,
    # the nested fn pointer become a freevar, making us
    # recompile every time.
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, since it's same kernel."

    doubleT_0_fn = wave_compile(options, double_transpose)
    assert (
        len(cache_manager.session_cache) == 2
    ), "Expected len == 2, since despite same core, it has different signature."


# This tests that when a free variable/constants
# is being used inside a nested kernel, it re-compiles.
# when we change the value and doesn't recompile
# when we use the same values.


@require_e2e
@require_cache
def testChangeFreeVarOfNestedFunction(tmp_path):
    reset_cache_manager(tmp_path)
    shape = [256, 256]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    def get_double_kernel(FREEVAR_VAL):
        def core(a):
            offset = tkl.Register[M, N, tkl.f16](FREEVAR_VAL)
            res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            double = res + res + offset
            tkw.write(double, a, elements_per_thread=ELEMS_PER_THREAD)

        @tkw.wave(constraints)
        def double_kernel(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
            core(a)

        return double_kernel

    cache_manager = get_cache_manager()
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
    )
    options = set_default_run_config(options)
    assert (
        len(cache_manager.session_cache) == 0
    ), "Expected len == 0, before any compilation of kernel."
    double_offset_1 = get_double_kernel(1.0)
    double_offset_1_fn = wave_compile(options, double_offset_1)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, after caching first kernel."

    double_offset_1_2nd = get_double_kernel(1.0)
    double_offset_1_2nd_fn = wave_compile(options, double_offset_1_2nd)
    assert (
        len(cache_manager.session_cache) == 1
    ), "Expected len == 1, since it's same kernel."

    double_offset_2 = get_double_kernel(2.0)
    double_offset_2_fn = wave_compile(options, double_offset_2)
    assert (
        len(cache_manager.session_cache) == 2
    ), "Expected len == 2, since despite same core, it has different signature."
