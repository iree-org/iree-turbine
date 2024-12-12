# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from torch.nn import functional as F
import math
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
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
    device_randint,
    to_default_device,
)
from iree.turbine.kernel.wave.constraints import MMAType
import os
import json
from torch.testing import assert_close, assert_allclose
from enum import Enum
import sympy


_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))
require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(), reason="Default device is not CDNA3"
)
# Whether to dump the generated MLIR module.
test_dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
default_test_shapes = {}
# Order of shapes: (B, M, N, K1, K2)
default_test_shapes["test_attention"] = [
    (8, 128, 128, 64, 256),
    (40, 1024, 64, 64, 1024),
]
default_test_shapes["test_attention"] += [
    perf_test(x) for x in default_test_shapes["test_attention"]
]

default_test_shapes["test_paged_decoding"] = [(32, 6, 0, 128, 1024)]

user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes[test_name]


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testChainedGemm(
    shape: tuple[int],
    enable_scheduling: bool,
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
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
        q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = device_randn(shape[0], shape[2], shape[4], dtype=torch.float16)
        output = device_zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        mb = chained_gemm(q, k, v, output)

        if test_dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, check_device=False)


# This test requires some more analysis on the index sequences between
# the two chained GEMMs.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testChainedGemmF8(
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, N: k}, outputs={B: i, N: k, M: j}
    )

    @tkw.wave(constraints)
    def chained_gemm_f8(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(qk_cast_reg, v_reg, acc, mfma_variant[1])
            return acc

        # repeat represents the results of the loop
        tkw.write(
            repeat, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
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
        v = device_randn(shape[0], shape[2], shape[4], dtype=torch.float16)
        output = device_zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        mb = chained_gemm_f8(q, k, v, output)

        if test_dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt_f8", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, atol=7e-5, rtol=2e-3, check_device=False)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testAttention(
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
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}

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
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
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

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        assert_allclose(output, torch_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
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

        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if "gfx94" in config["target"]:
            assert_allclose(output, torch_ref, atol=2e-3, rtol=5e-3)
        else:
            # TODO: Determine why the error is higher on gfx90.
            assert_allclose(output, torch_ref, atol=3e-3, rtol=8e-1)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
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
        if test_dump_generated_mlir:
            filename = f"wave_attention_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())
        rmse = torch.sqrt(torch.mean(torch.square(output - torch_ref)))
        assert rmse <= 0.006


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("dynamic_dims", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    ],
)
def testFlashDecoding(
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

    class Phase(Enum):
        QK = (0,)
        SOFTMAX_V = (1,)

    def get_constraints(phase: Phase) -> list[tkw.Constraint]:
        if mfma_variant == MMAType.F32_16x16x16_F16:
            Mvec = 16
            Nvec = 16
        if mfma_variant == MMAType.F32_32x32x8_F16:
            Mvec = 32
            Nvec = 32
        ratio_m = 2
        ratio_n = 2
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
        constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
        if phase == Phase.QK:
            constraints += [tkw.WorkgroupConstraint(K2, BLOCK_K2, 1)]
            constraints += [tkw.WaveConstraint(K2, BLOCK_K2 / ratio_n)]
            vector_shapes = {B: 0, M: Mvec, K2: Nvec}
        else:
            constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
            constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
            constraints += [tkw.WaveConstraint(N, BLOCK_N / ratio_n)]
            vector_shapes = {B: 0, M: Mvec, N: Nvec}
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=64,
                waves_per_block=(ratio_m, ratio_n, 1),
                mma_type=mfma_variant,
                vector_shapes=vector_shapes,
            )
        ]
        if dynamic_dims:
            constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]
        return constraints

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    # The first kernel computes K @ Q.T.
    @tkw.wave(get_constraints(Phase.QK))
    def qk_kernel(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(k_reg, q_reg, c_reg)
        x_j = tkw.permute(acc, target_shape=[B, M, K2])
        tkw.write(x_j, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    # The second kernel computes the softmax and V @ softmax(K @ Q.T).
    @tkw.wave(get_constraints(Phase.SOFTMAX_V))
    def softmax_v_kernel(
        qk: tkl.Memory[B, M, K2, ADDRESS_SPACE, tkl.f32],
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
            x_j = tkw.read(qk, elements_per_thread=STORE_ELEMS_PER_THREAD)
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
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
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

    torch.manual_seed(0)
    q = device_randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = device_randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    qk = device_zeros(shape[0], shape[1], shape[4], dtype=torch.float32)
    v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])

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
        mb_qk = qk_kernel(q * dk_sqrt * log2e, k, qk)

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
        # TODO: Add variant of non-transposed V attention kernel.
        mb_sv = softmax_v_kernel(qk, v.permute([0, 2, 1]), output)

    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )

    if test_dump_generated_mlir:
        filename = f"wave_qk_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())
        filename = f"wave_softmax_v_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_sv.module_op.get_asm())

    assert_allclose(output, torch_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_paged_decoding"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("dynamic_dims", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
    ],
)
def testPagedFlashDecoding(
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

    class Phase(Enum):
        QK = (0,)
        SOFTMAX_V = (1,)

    def get_constraints(phase: Phase) -> list[tkw.Constraint]:
        if mfma_variant == MMAType.F32_16x16x16_F16:
            Mvec = 16
            Nvec = 16
        if mfma_variant == MMAType.F32_32x32x8_F16:
            Mvec = 32
            Nvec = 32
        ratio_m = 1
        ratio_n = 2
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
        # constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
        if phase == Phase.QK:
            constraints += [tkw.WorkgroupConstraint(K2, BLOCK_K2, 1)]
            constraints += [tkw.WaveConstraint(K2, BLOCK_K2 / ratio_n)]
            vector_shapes = {B: 0, M: Mvec, K2: Nvec}
        else:
            constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
            constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
            constraints += [tkw.WaveConstraint(N, BLOCK_N / ratio_n)]
            vector_shapes = {B: 0, M: Mvec, N: Nvec}
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=64,
                waves_per_block=(ratio_m, ratio_n, 1),
                mma_type=mfma_variant,
                vector_shapes=vector_shapes,
            )
        ]
        if dynamic_dims:
            constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]
        return constraints

    # Shape of logical to physical mapping table.
    R0 = 4097
    R1 = 8196
    BATCH = shape[0]
    NUM_HEADS = shape[1]
    HEAD_DIM = shape[3]
    SEQ_LEN = shape[4]
    HEAD_TILE_SIZE = 16
    BK = BATCH * NUM_HEADS

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    d0 = tkw.IndexMapping.dynamic_val(0)

    # Load a specific element from the request_to_tokens matrix.
    # The request_to_tokens matrix has shape [R0, R1] and we are loading a single element.
    request_to_tokens_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: d0, K2: j},
        outputs={B: i, K2: j},
        dynamic_val_mappings={B: i},
    )

    # Broadcast the offset along the batch dimension.
    k_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={
            B: d0 + WORKGROUP_0 / sympy.ceiling(NUM_HEADS / HEAD_TILE_SIZE),
            K2: sympy.Integer(0),
            K1: k,
        },
        outputs={B: i, K2: j, K1: k},
        dynamic_val_mappings={B: i // LOAD_ELEMS_PER_THREAD, K2: j},
    )

    # Broadcast the offset along the batch dimension.
    output_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M: j, K2: k},
        outputs={B: d0, M: j, K2: k},
        dynamic_val_mappings={B: i // STORE_ELEMS_PER_THREAD},
    )

    q_layout = {"stride": (SEQ_LEN, K1, 1)}
    k_layout = {"stride": (K1, K1, 1), "shape": (BK, 1, K1)}
    req_to_tokens_layout = {"stride": (R1, 1), "shape": (R0, R1)}

    # The first kernel computes K @ Q.T.
    @tkw.wave(get_constraints(Phase.QK))
    def qk_kernel(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16, q_layout],
        k_cache: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16, k_layout],
        request_to_tokens: tkl.Memory[
            B, K2, GLOBAL_ADDRESS_SPACE, tkl.i32, req_to_tokens_layout
        ],
        request_indices: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32],
        request_offsets: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32],
        output: tkl.Memory[B, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
        req_idx_reg = tkw.read(request_indices, elements_per_thread=1)
        q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        k_offsets = tkw.read(
            request_to_tokens,
            elements_per_thread=1,
            mapping=request_to_tokens_mapping,
            mapping_dynamic_vals=(req_idx_reg,),
        )
        k_reg = tkw.read(
            k_cache,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=k_mapping,
            mapping_dynamic_vals=(k_offsets,),
        )
        acc = tkw.mma(k_reg, q_reg, c_reg)
        x_j = tkw.permute(acc, target_shape=[B, M, K2])
        req_off_reg = tkw.read(request_offsets, elements_per_thread=1)
        tkw.write(
            x_j,
            output,
            elements_per_thread=STORE_ELEMS_PER_THREAD,
            mapping=output_mapping,
            mapping_dynamic_vals=(req_off_reg,),
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 16,
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

    torch.manual_seed(0)

    # Construct query matrix with correct stride.
    desired_shape = (BATCH, NUM_HEADS, HEAD_DIM)
    desired_stride = (SEQ_LEN, HEAD_DIM, 1)
    space_required = sum([a * b for a, b in zip(desired_shape, desired_stride)])
    q = device_randn(space_required, dtype=torch.float16)
    q = torch.as_strided(q, desired_shape, desired_stride)

    # Construct synthetic page tables for key matrix.
    total_entries = BATCH * NUM_HEADS
    request_to_tokens = device_zeros(R0, R1, dtype=torch.int32)
    request_to_tokens[0:BATCH, 0:SEQ_LEN] = (
        torch.arange(BATCH * SEQ_LEN).reshape(BATCH, SEQ_LEN) + 1
    )
    request_to_tokens[0:BATCH, SEQ_LEN] = torch.arange(
        BATCH * SEQ_LEN, BATCH * (SEQ_LEN + 1)
    )
    request_indices = to_default_device(torch.arange(BATCH, dtype=torch.int32))
    request_offsets = to_default_device(torch.arange(BATCH, dtype=torch.int32))
    desired_stride = (HEAD_DIM, HEAD_DIM, 1)
    desired_shape = (BATCH * NUM_HEADS, 1, HEAD_DIM)
    space_required = sum([a * b for a, b in zip(desired_shape, desired_stride)])
    k_cache = device_randn(space_required, dtype=torch.float16)
    k_cache = torch.as_strided(k_cache, desired_shape, desired_stride)

    # def extract_page_table_entries(request_to_tokens, request_indices, request_offsets):
    #    entries = []
    #    for request in request_indices:
    #        token_idx = request_to_tokens[request]
    #        entries.append(k_cache[token_idx])
    #        # TODO: Broadcast entries to shape [B, K2, K1].
    #    return torch.cat(entries, dim=0)

    # k = extract_page_table_entries(request_to_tokens, request_indices, request_offsets)
    qk = device_zeros(shape[0], shape[1], shape[4], dtype=torch.float32)

    # v = device_randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    # output = device_zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape[3])

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
        mb_qk = qk_kernel(
            q * dk_sqrt * log2e,
            k_cache,
            request_to_tokens,
            request_indices,
            request_offsets,
            qk,
        )

    breakpoint()
    # torch_ref = torch.matmul(q, k.permute([0, 2, 1])) * dk_sqrt * log2e
    # assert_allclose(qk, torch_ref.permute([0, 2, 1]))

    # with tk.gen.TestLaunchContext(
    #    hyperparams,
    #    canonicalize=True,
    #    run=True,
    #    run_bench=run_bench,
    #    run_config=config,
    #    schedule=enable_scheduling,
    #    use_scheduling_barriers=enable_scheduling_barriers,
    #    dynamic_symbols=dynamic_symbols,
    #    dynamic_symbols_map=dynamic_symbols_map,
    # ):
    #    # TODO: Add variant of non-transposed V attention kernel.
    #    mb_sv = softmax_v_kernel(qk, v.permute([0, 2, 1]), output)

    # torch_ref = torch.nn.functional.scaled_dot_product_attention(
    #    q, k, v, attn_mask=None
    # )

    if test_dump_generated_mlir:
        filename = f"wave_qk_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(mb_qk.module_op.get_asm())
        # filename = f"wave_softmax_v_kernel_{'x'.join(map(str, shape))}.mlir"
        # with open(filename, "w") as f:
        #    f.write(mb_sv.module_op.get_asm())

    # assert_allclose(output, torch_ref)
