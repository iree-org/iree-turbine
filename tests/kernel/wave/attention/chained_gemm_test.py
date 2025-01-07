# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
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


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("chained_gemm"))
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

        if dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("chained_gemm"))
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

        if dump_generated_mlir:
            filename = f"wave_cgemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[2], shape[1], dtype=torch.float32)
        generate_iree_ref(
            "chain_mmt_f8", [q, k, v], [iree_ref], config, run_bench=run_bench
        )
        assert_close(output, iree_ref, atol=7e-5, rtol=2e-3, check_device=False)
