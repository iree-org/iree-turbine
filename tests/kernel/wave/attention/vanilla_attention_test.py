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
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import MMAType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def testAttention(
    shape: tuple[int],
    enable_scheduling: bool,
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
    ) = get_vanilla_attention_kernel(shape, mfma_variant, dynamic_dims)
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(q_shape, dtype=torch.float16)
        k = device_randn(k_shape, dtype=torch.float16)
        v = device_randn(v_shape, dtype=torch.float16)
        output = device_zeros(o_shape, dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape.head_size)
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )

        if dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionBias(
    shape: tuple[int],
    enable_scheduling: bool,
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
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            x_j = x_j + bias_reg
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
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

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
        mb = base_attention_bias(
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
                f.write(mb.module_op.get_asm())

        if "gfx94" in config["target"]:
            assert_close(output, torch_ref, atol=2e-3, rtol=5e-3, check_dtype=False)
        else:
            # TODO: Determine why the error is higher on gfx90.
            assert_close(output, torch_ref, atol=3e-3, rtol=8e-1, check_dtype=False)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttentionSoftCap(
    shape: tuple[int],
    enable_scheduling: bool,
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
            x_j = soft_cap * tkw.tanh(x_j / soft_cap)
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
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

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
        mb = base_attention_softcap(
            q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output
        )
        k_t = k.transpose(-1, -2)
        a = torch.matmul(q, k_t) * dk_sqrt
        a = softcap_val * torch.tanh(a / softcap_val)
        a = F.softmax(a, dim=-1)
        torch_ref = torch.matmul(a, v)

        if dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if "gfx94" in config["target"]:
            assert_close(output, torch_ref, atol=2e-3, rtol=5e-3, check_dtype=False)
        else:
            # TODO: Determine why the error is higher on gfx90.
            assert_close(output, torch_ref, atol=3e-3, rtol=8e-1, check_dtype=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("attention"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testAttentionF8(
    shape: tuple[int], enable_scheduling: bool, mfma_variant: tuple[MMAType], request
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
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
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
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f8, new_acc, mfma_variant[1])
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
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
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape[3])
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )
        if dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())
        rmse = torch.sqrt(torch.mean(torch.square(output - torch_ref)))
        assert rmse <= 0.006
