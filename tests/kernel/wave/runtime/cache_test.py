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
import pytest
import torch
from torch.testing import assert_close
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw

from iree.turbine.kernel.wave.cache import (
    is_cache_enabled,
    get_cache_manager,
    reset_cache_manager,
    WaveCache,
)
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_arch,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import Constraint, MMAType
import os

require_e2e = pytest.mark.require_e2e

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
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
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
def testSameConfig(request):
    reset_cache_manager()
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
    config = get_default_run_config()
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

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
    with tk.gen.TestLaunchContext(
        copy.deepcopy(hyperparams),
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
    ):
        assert (
            len(cache_manager.session_cache) == 0
        ), "Expected to start runtime with no cache."

        # First run/call to kernel, this should compile from scratch.
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
        assert isinstance(
            mb, tk.compiler.builder.ModuleBuilder
        ), "Expected first call to not be cached."
        assert (
            len(cache_manager.session_cache) == 1
        ), "Expected len == 1, after caching first kernel."

        # Subsequent run/call to kernel, this should be using cached.
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        cached_kernel = base_attention(
            q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output
        )
        assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
        assert (
            len(cache_manager.session_cache) == 1
        ), "Expected to keep size of cache because we reuse same kernel."
        assert isinstance(
            cached_kernel, WaveCache
        ), "Expected subsequent call to be cached."


@require_e2e
@require_cache
def testDifferentDynamicSameBlock(request):
    reset_cache_manager()
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
    config = get_default_run_config()
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

    dynamic_symbols = [M, N, B, K2]
    dynamic_sym_shape0 = {M: shape_0[1], N: shape_0[2], B: shape_0[0], K2: shape_0[4]}

    cache_manager = get_cache_manager()

    # First run/call to kernel, this should compile from scratch.
    with tk.gen.TestLaunchContext(
        copy.deepcopy(hyperparams),
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_sym_shape0,
    ):
        assert (
            len(cache_manager.session_cache) == 0
        ), "Expected to start runtime with no cache."

        torch.manual_seed(0)
        q_shape_0 = device_randn(
            shape_0[0], shape_0[1], shape_0[3], dtype=torch.float16
        )
        k_shape_0 = device_randn(
            shape_0[0], shape_0[4], shape_0[3], dtype=torch.float16
        )
        v_shape_0 = device_randn(
            shape_0[0], shape_0[4], shape_0[2], dtype=torch.float16
        )
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape_0[3])
        torch_ref_shape_0 = torch.nn.functional.scaled_dot_product_attention(
            q_shape_0, k_shape_0, v_shape_0, attn_mask=None
        ).to(torch.float32)
        output_shape_0 = device_zeros(
            shape_0[0], shape_0[1], shape_0[2], dtype=torch.float32
        )
        mb = base_attention(
            q_shape_0 * dk_sqrt * log2e,
            k_shape_0,
            v_shape_0.permute([0, 2, 1]),
            output_shape_0,
        )
        assert_close(output_shape_0, torch_ref_shape_0, atol=1e-3, rtol=1e-3)
        assert isinstance(
            mb, tk.compiler.builder.ModuleBuilder
        ), "Expected first call to not be cached."
        assert (
            len(cache_manager.session_cache) == 1
        ), "Expected len == 1, after caching first kernel."

    # Despite having different problem size, since we use exact same
    # block size we should be able to use cache.
    shape_1 = (2, 128, 64, 64, 128)
    dynamic_sym_shape1 = {M: shape_1[1], N: shape_1[2], B: shape_1[0], K2: shape_1[4]}
    with tk.gen.TestLaunchContext(
        copy.deepcopy(hyperparams),
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_sym_shape1,
    ):
        torch.manual_seed(0)
        q_shape_1 = device_randn(
            shape_1[0], shape_1[1], shape_1[3], dtype=torch.float16
        )
        k_shape_1 = device_randn(
            shape_1[0], shape_1[4], shape_1[3], dtype=torch.float16
        )
        v_shape_1 = device_randn(
            shape_1[0], shape_1[4], shape_1[2], dtype=torch.float16
        )
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
        cached_kernel = base_attention(
            q_shape_1 * dk_sqrt * log2e,
            k_shape_1,
            v_shape_1.permute([0, 2, 1]),
            output_shape_1,
        )
        assert_close(output_shape_1, torch_ref_shape_1, atol=1e-3, rtol=1e-3)
        assert (
            len(cache_manager.session_cache) == 1
        ), "Expected to keep size of cache because we reuse same kernel."
        assert isinstance(
            cached_kernel, WaveCache
        ), "Expected subsequent call to be cached."


@require_e2e
@require_cache
def testSameSizeDifferentBlock(request):
    reset_cache_manager()
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
    config = get_default_run_config()
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

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
    with tk.gen.TestLaunchContext(
        copy.deepcopy(hyperparams),
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
    ):
        assert (
            len(cache_manager.session_cache) == 0
        ), "Expected to start runtime with no cache."

        # First run/call to kernel, this should compile from scratch.
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        mb_config_0 = base_attention(
            q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output
        )
        assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
        assert isinstance(
            mb_config_0, tk.compiler.builder.ModuleBuilder
        ), "Expected first call to not be cached."
        assert (
            len(cache_manager.session_cache) == 1
        ), "Expected len == 1, after caching first kernel."

    # Subsequent run/call to kernel, this trigger recompile because we use
    # a different block size/config.
    hyperparams[BLOCK_N] = 32
    hyperparams[BLOCK_K2] = 32
    with tk.gen.TestLaunchContext(
        copy.deepcopy(hyperparams),
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        compile_config=compile_config,
    ):
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        mb_config_1 = base_attention(
            q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output
        )
        assert_close(output, torch_ref, atol=1e-3, rtol=1e-3)
        assert (
            len(cache_manager.session_cache) == 2
        ), "Expected cache size to increment, because we use different block size/config."
        assert isinstance(
            mb_config_1, tk.compiler.builder.ModuleBuilder
        ), "Expected subsequent call to be cached."
