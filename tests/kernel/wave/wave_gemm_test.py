# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
import os
import json
from torch.testing import assert_close

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))
require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")
# Whether to dump the generated MLIR module.
test_dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))


default_test_shapes = [
    (2048, 10240, 1280, 128, 320, 32, 2, 2, 2, 2, 2, 2, 1, 1, 2),
    (2048, 1280, 1280, 64, 64, 64, 2, 2, 1, 2, 1, 1, 1, 1, 2),
    (2048, 1280, 5120, 128, 80, 128, 4, 1, 1, 4, 2, 2, 1, 1, 2),
    (128, 1280, 2048, 64, 64, 128, 2, 2, 1, 8, 2, 2, 1, 1, 2),
    (8192, 5120, 640, 128, 128, 32, 2, 2, 1, 4, 2, 2, 1, 1, 2),
]

perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)

default_test_shapes += [perf_test(x) for x in default_test_shapes]

user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes


@require_e2e
@pytest.mark.parametrize("params", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
def testGemm(params: tuple[int], enable_scheduling: bool, request):
    (
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        ratio_m,
        ratio_n,
        waves_per_eu,
        mma_units,
        shared_units,
        global_units,
        delay_mma,
        delay_shared,
        delay_global,
    ) = params
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / ratio_n)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(ratio_m, ratio_n, 1)
        )
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        READ_SHARED_DELAY: delay_shared,
        WRITE_SHARED_DELAY: delay_shared,
        READ_GLOBAL_DELAY: delay_global,
        WRITE_GLOBAL_DELAY: delay_global,
        MMA_DELAY: delay_mma,
        SHARED_MEMORY_UNITS: shared_units,
        GLOBAL_MEMORY_UNITS: global_units,
        MMA_UNITS: mma_units,
    }
    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}
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
        a = torch.randn(m, k, dtype=torch.float16)
        b = torch.randn(n, k, dtype=torch.float16)
        c = torch.zeros(m, n, dtype=torch.float32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            shape = [m, n, k]
            filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
        iree_ref = torch.zeros(m, n, dtype=torch.float32)
        generate_iree_ref("mmt", [a, b], [iree_ref], config, run_bench=run_bench)
        assert_close(c, iree_ref)
