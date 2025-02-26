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
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_arch,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_randint,
    device_zeros,
)
from iree.turbine.kernel.lang import DataType
from iree.turbine.kernel.wave.constraints import MMAType
import os
import json
from torch.testing import assert_close
from enum import Enum

require_e2e = pytest.mark.require_e2e
require_cdna2 = pytest.mark.skipif(
    "gfx90" not in get_default_arch(), reason="Default device is not CDNA2"
)
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
default_test_shapes["test_gemm"] = [
  # (1024, 5120, 640),
  # (8, 4, 8),
    (16, 8, 16),
  # (2048, 10240, 1280),
  # (4096, 20480, 2560),
]
#default_test_shapes["test_gemm"] += [
#    perf_test(x) for x in default_test_shapes["test_gemm"]
#]
default_test_shapes["test_batched_gemm"] = [(8, 256, 128, 192), (32, 1024, 512, 768)]


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
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize("dynamic_dims", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
    #   MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testGemm(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 1, 1), mma_type=mfma_variant,
           #vector_shapes={K: 0, M: 16, N: 16},
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the reduction dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

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
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        d: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
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
            d_reg = tkw.read(d, elements_per_thread=STORE_ELEMS_PER_THREAD)
            return acc * d_reg

        # repeat represents the results of the loop
        casted = tkw.cast(repeat, tkl.f16)
        tkw.write(casted, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 32,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
        dynamic_symbols_map[K] = hyperparams[K]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(K)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[K]

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
      # a = device_randn(shape[0], shape[2], dtype=torch.float16)
      # b = device_randn(shape[1], shape[2], dtype=torch.float16)
      # c = device_zeros(shape[0], shape[1], dtype=torch.float32)
      # d = device_randn(shape[0], shape[1], dtype=torch.float32)

        rows, cols = shape[0], shape[2]
        a = torch.arange(0, rows * cols, dtype=torch.float16, device="cuda").reshape(rows, cols)
        rows, cols = shape[1], shape[2]
        b = torch.arange(1, rows * cols + 1, dtype=torch.float16, device="cuda").reshape(rows, cols)
        c = device_zeros(shape[0], shape[1], dtype=torch.float16)
        rows, cols = shape[0], shape[1]
        d = torch.full((rows, cols), 2, dtype=torch.float32, device="cuda")
        mb = gemm(a, b, c, d)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op)

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
       #iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.float32)
       #generate_iree_ref("mmt", [a, b], [iree_ref], config, run_bench=run_bench)
       #assert_close(c, iree_ref, check_device=False)
        torch_ref = torch.matmul(a, b.transpose(-1, -2)) * d
        print(torch_ref)
        print(c)
        assert_close(c, torch_ref, atol=3e-3, rtol=8e-3, check_dtype=False)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_32x32x16_K8_F16,
        MMAType.F32_16x16x32_K8_F16,
    ],
)
def testVMFMAGemm(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the reduction dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

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
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
        dynamic_symbols_map[K] = hyperparams[K]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(K)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[K]

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
        a = device_randn(shape[0], shape[2], dtype=torch.float16)
        b = device_randn(shape[1], shape[2], dtype=torch.float16)
        c = device_zeros(shape[0], shape[1], dtype=torch.float32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
        iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        generate_iree_ref("mmt", [a, b], [iree_ref], config, run_bench=run_bench)
        assert_close(c, iree_ref, atol=2e-4, rtol=3e-4, check_device=False)


@require_e2e
@require_cdna2
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize("dynamic_dims", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.I32_16x16x16_I8,
        MMAType.I32_32x32x8_I8,
    ],
)
def testCDNA2IntGemm(
    shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the reduction dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.i32]:
            # a_reg: tkw.Register[M, K, tkl.i8]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.i8]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.i32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
        dynamic_symbols_map[K] = hyperparams[K]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(K)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[K]

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
        randint_hi = 4
        a = device_randint(randint_hi, (shape[0], shape[2]), dtype=torch.int8)
        b = device_randint(randint_hi, (shape[1], shape[2]), dtype=torch.int8)
        c = device_zeros(shape[0], shape[1], dtype=torch.int32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
        iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.int32)
        generate_iree_ref("mmt", [a, b], [iree_ref], config, run_bench=run_bench)
        assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.I32_16x16x32_I8,
        MMAType.I32_32x32x16_I8,
    ],
)
def testCDNA3IntGemm(
    shape: tuple[int], enable_scheduling: bool, mfma_variant: MMAType, request
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.i32]:
            # a_reg: tkw.Register[M, K, tkl.i8]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.i8]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.i32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
        randint_hi = 4
        a = device_randint(randint_hi, (shape[0], shape[2]), dtype=torch.int8)
        b = device_randint(randint_hi, (shape[1], shape[2]), dtype=torch.int8)
        c = device_zeros(shape[0], shape[1], dtype=torch.int32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}_f8.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
        iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.int32)
        generate_iree_ref("mmt", [a, b], [iree_ref], config, run_bench=run_bench)
        assert_close(c, iree_ref, check_device=False)


@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x32_F8,
        MMAType.F32_16x16x32_K4_F8,
        MMAType.F32_32x32x16_F8,
        MMAType.F32_32x32x16_K4_F8,
    ],
)
def testF8Gemm(
    shape: tuple[int], enable_scheduling: bool, mfma_variant: MMAType, request
):
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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            a_reg = tkw.cast(a_reg, tkl.f8e4m3fnuz)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.cast(b_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
        a = device_randn(shape[0], shape[2], dtype=torch.float16)
        b = device_randn(shape[1], shape[2], dtype=torch.float16)
        c = device_zeros(shape[0], shape[1], dtype=torch.float32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}_f8.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )
        iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        generate_iree_ref("mmt_f8", [a, b], [iree_ref], config, run_bench=run_bench)
        assert_close(c, iree_ref, atol=3e-5, rtol=3e-4, check_device=False)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_batched_gemm"))
@pytest.mark.parametrize("enable_scheduling", [False, True])
def testBatchedGemmGradV(shape: tuple[int], enable_scheduling: bool, request):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
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
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), vector_shapes={B: 0}
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    do_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, M: j, K: k}, outputs={B: i, M: j, K: k}
    )
    a_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, K: k}, outputs={B: i, N: j, K: k}
    )
    dv_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, N: j, M: k}
    )

    @tkw.wave(constraints)
    def batched_gemm_grad_v(
        do: tkl.Memory[K, B, M, ADDRESS_SPACE, tkl.f16],
        a: tkl.Memory[B, K, N, ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[N, B, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        dv_reg = tkl.Register[B, N, M, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[dv_reg])
        def repeat(
            acc: tkl.Register[B, N, M, tkl.f32]
        ) -> tkl.Register[B, N, M, tkl.f32]:
            do_reg = tkw.read(do, mapping=do_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            a_reg = tkw.read(a, mapping=a_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, do_reg, acc)
            return acc

        tkw.write(repeat, dv, mapping=dv_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K: shape[3],
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
        do = device_randn(shape[3], shape[0], shape[1], dtype=torch.float16)
        a = device_randn(shape[0], shape[3], shape[2], dtype=torch.float16)
        dv = device_zeros(shape[2], shape[0], shape[1], dtype=torch.float32)
        mb = batched_gemm_grad_v(do, a, dv)

        if test_dump_generated_mlir:
            filename = f"wave_batched_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op)

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )

        torch_ref = torch.matmul(a.transpose(-1, -2), do.transpose(-2, -3)).transpose(-2, -3)
        assert_close(dv, torch_ref, atol=2e-3, rtol=5e-3, check_dtype=False)


@require_e2e
@pytest.mark.parametrize("shape", [(1, 256, 256, 4, 32), (1, 512, 256, 8, 8)])
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_32x32x8_F16,
    ],
)
@pytest.mark.parametrize("dtype", [tkl.f16, tkl.bf16])
def testBatchedGemm(shape: tuple[int],
    enable_scheduling: bool,
    mfma_variant: MMAType,
    dtype: DataType,
    request
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    batch, n, seq_len, heads, dim = shape
    q_seq_len = seq_len
    kv_seq_len = seq_len
  # head_dim = dim
  # v_dim = dim

    # Input sizes
    B = tkl.sym.B
    BN = tkl.sym.BN
    H = tkl.sym.H
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BN = tkl.sym.BLOCK_BN
    BLOCK_H = tkl.sym.BLOCK_H
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
  # constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(K, BLOCK_K, 0)]
  # constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
  # constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 2)]
  # constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 3)]
  # constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 4)]
  # constraints += [tkw.TilingConstraint(M, BLOCK_M)]
  # constraints += [tkw.WaveConstraint(K, BLOCK_K / 2)]
  # constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
  # constraints += [
  #     tkw.HardwareConstraint(
  #         threads_per_wave=64, waves_per_block=(2, 1, 1),
  #         mma_type=mfma_variant,
  #         vector_shapes={B: 0, BN: 0, H: 0, K: 32, N: 32},
  #     )
  # ]

  # i = tkw.IndexMapping.iterator(0)
  # j = tkw.IndexMapping.iterator(1)
  # k = tkw.IndexMapping.iterator(2)
  # l = tkw.IndexMapping.iterator(3)
  # m = tkw.IndexMapping.iterator(4)
  # do_mapping = tkw.IndexMapping(
  #     num_iterators=5, inputs={B: i, BN: j, H: k, K: l, M: m}, outputs={B: i, BN: j, H: k, K: l, M: m}
  # )
  # a_mapping = tkw.IndexMapping(
  #     num_iterators=5, inputs={B: i, BN: j, H: k, N: l, M: m}, outputs={B: i, BN: j, H: k, N: l, M: m}
  # )
  # dv_mapping = tkw.IndexMapping(
  #     num_iterators=5, inputs={B: i, BN: j, H: k, N: l, K: m}, outputs={B: i, BN: j, H: k, N: l, K: m}
  # )

  # @tkw.wave(constraints)
  # def batched_gemm_grad_v(
  #     do: tkl.Memory[B, BN, M, H, K, GLOBAL_ADDRESS_SPACE, dtype],
  #     a: tkl.Memory[B, BN, H, M, N, GLOBAL_ADDRESS_SPACE, dtype],
  #     dv: tkl.Memory[B, BN, N, H, K, GLOBAL_ADDRESS_SPACE, dtype],
  # ):
  #     dv_reg = tkl.Register[B, BN, H, N, K, tkl.f32](0.0)

  #     @tkw.reduction(M, init_args=[dv_reg])
  #     def repeat(
  #         acc: tkl.Register[B, BN, H, N, K, tkl.f32]
  #     ) -> tkl.Register[B, BN, H, N, K, tkl.f32]:
  #         do_reg = tkw.read(do, mapping=do_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
  #         if dtype == tkl.bf16:
  #             do_reg = tkw.cast(tkw.cast(do_reg, tkl.f32), tkl.f16)
  #         a_reg = tkw.read(a, mapping=a_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
  #         if dtype == tkl.bf16:
  #             a_reg = tkw.cast(tkw.cast(a_reg, tkl.f32), tkl.f16)
  #         acc = tkw.mma(a_reg, do_reg, acc)
  #         return acc

  #     casted = tkw.cast(repeat, dtype)
  #     tkw.write(casted, dv, mapping=dv_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

  # constraints.clear()
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
  # constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 2)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 3)]
    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 4)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, BN: 0, H: 0, M: 32, N: 32},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    m = tkw.IndexMapping.iterator(4)
    do_mapping = tkw.IndexMapping(
        num_iterators=5, inputs={B: i, BN: j, H: k, M: l, K: m}, outputs={B: i, BN: j, H: k, M: l, K: m}
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=5, inputs={B: i, BN: j, H: k, N: l, K: m}, outputs={B: i, BN: j, H: k, N: l, K: m}
    )

    @tkw.wave(constraints)
    def batched_gemm_grad_a(
        do: tkl.Memory[B, BN, M, H, K, GLOBAL_ADDRESS_SPACE, dtype],
        a: tkl.Memory[B, BN, H, M, N, GLOBAL_ADDRESS_SPACE, dtype],
        v: tkl.Memory[B, BN, N, H, K, GLOBAL_ADDRESS_SPACE, dtype],
        da: tkl.Memory[B, BN, H, M, N, GLOBAL_ADDRESS_SPACE, dtype],
    ):
        da_reg = tkl.Register[B, BN, H, M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[da_reg])
        def repeat(
            acc: tkl.Register[B, BN, H, M, N, tkl.f32]
        ) -> tkl.Register[B, BN, H, M, N, tkl.f32]:
            do_reg = tkw.read(do, mapping=do_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            if dtype == tkl.bf16:
                do_reg = tkw.cast(tkw.cast(do_reg, tkl.f32), tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            if dtype == tkl.bf16:
                v_reg = tkw.cast(tkw.cast(v_reg, tkl.f32), tkl.f16)
            grad_a = tkw.mma(do_reg, v_reg, acc)
            # WTF WHY NOT LOAD_ELEMS_PER_THREAD
            a_reg = tkw.read(a, elements_per_thread=STORE_ELEMS_PER_THREAD)
            a_reg = tkw.cast(a_reg, tkl.f32)
            acc = grad_a * a_reg
            return acc

        casted = tkw.cast(repeat, dtype)
        tkw.write(casted, da, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        # WTF why vector sizes affect correctness?
      # LOAD_ELEMS_PER_THREAD: 4,
      # STORE_ELEMS_PER_THREAD: 4,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        # WTF why tile sizes affect correctness?
      # BLOCK_B: 1,
      # BLOCK_BN: 1,
      # BLOCK_M: 64,
      # BLOCK_N: 64,
      # BLOCK_H: 64,
      # BLOCK_K: 32,

        BLOCK_B: 1,
        BLOCK_BN: 1,
        BLOCK_M: 64,
        BLOCK_N: 32,
        BLOCK_H: 1,
        BLOCK_K: 32,

        B: batch,
        BN: n,
        M: q_seq_len,
        N: kv_seq_len,
        H: heads,
        K: dim,
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
        if dtype == tkl.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        do = device_randn(batch, n, q_seq_len, heads, dim, dtype=torch_dtype)
        a = device_randn(batch, n, heads, q_seq_len, kv_seq_len, dtype=torch_dtype)
        dv = device_zeros(batch, n, kv_seq_len, heads, dim, dtype=torch_dtype)
        v = device_randn(batch, n, kv_seq_len, heads, dim, dtype=torch_dtype)
        da = device_zeros(batch, n, heads, q_seq_len, kv_seq_len, dtype=torch_dtype)
      # mb_grad_v = batched_gemm_grad_v(do, a, dv)
        mb = batched_gemm_grad_a(do, a, v, da)

        if test_dump_generated_mlir:
            filename = f"wave_batched_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op)

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )

      # dv_ref = torch.matmul(a.transpose(-1, -2), do.transpose(-2, -3)).transpose(-2, -3)
      # assert_close(dv, dv_ref, atol=3e-3, rtol=8e-3, check_dtype=False)

        grad_a = torch.matmul(do.transpose(-2, -3), v.transpose(-2, -3).transpose(-1, -2))
        print("RES ", da)
        print("REF ", grad_a * a)
        assert_close(da, grad_a * a, atol=3e-3, rtol=8e-3, check_dtype=False)


#@require_e2e
#@pytest.mark.parametrize("shape", [(1, 256, 256, 4, 32), (1, 512, 256, 8, 8)])
#@pytest.mark.parametrize("enable_scheduling", [False])
#@pytest.mark.parametrize(
#    "mfma_variant",
#    [
#        MMAType.F32_32x32x8_F16,
#    ],
#)
#@pytest.mark.parametrize("dtype", [tkl.f16, tkl.bf16])
#def testBatchedGemmAllGrads(shape: tuple[int],
#    enable_scheduling: bool,
#    mfma_variant: MMAType,
#    dtype: DataType,
#    request
#):
#    run_bench = request.config.getoption("--runperf")
#    dump_perf = request.config.getoption("--dump-perf-files-path")
#
#    batch, n, seq_len, heads, dim = shape
#    q_seq_len = seq_len
#    kv_seq_len = seq_len
#  # head_dim = dim
#  # v_dim = dim
#
#    # Input sizes
#    B = tkl.sym.B
#    BN = tkl.sym.BN
#    H = tkl.sym.H
#    M = tkl.sym.M
#    N = tkl.sym.N
#    K = tkl.sym.K
#    # Workgroup tile sizes
#    BLOCK_B = tkl.sym.BLOCK_B
#    BLOCK_BN = tkl.sym.BLOCK_BN
#    BLOCK_H = tkl.sym.BLOCK_H
#    BLOCK_M = tkl.sym.BLOCK_M
#    BLOCK_N = tkl.sym.BLOCK_N
#    BLOCK_K = tkl.sym.BLOCK_K
#    # Address space (for GPU, shared(1) or global(0))
#    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
#    # Other hyperparameters
#    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
#    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
#
#    # Expose user-constraints
#    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(K, BLOCK_K, 0)]
#    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
#    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 2)]
#    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 3)]
#    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 4)]
#    constraints += [tkw.TilingConstraint(M, BLOCK_M)]
#    constraints += [tkw.WaveConstraint(K, BLOCK_K / 2)]
#    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
#
#    constraints += [
#        tkw.HardwareConstraint(
#            threads_per_wave=64, waves_per_block=(2, 1, 1),
#            mma_type=mfma_variant,
#            vector_shapes={B: 0, BN: 0, H: 0, K: 32, N: 32},
#        )
#    ]
#
#    i = tkw.IndexMapping.iterator(0)
#    j = tkw.IndexMapping.iterator(1)
#    k = tkw.IndexMapping.iterator(2)
#    l = tkw.IndexMapping.iterator(3)
#    m = tkw.IndexMapping.iterator(4)
#    do_mapping = tkw.IndexMapping(
#        num_iterators=5, inputs={B: i, BN: j, H: k, K: l, M: m}, outputs={B: i, BN: j, H: k, K: l, M: m}
#    )
#    a_mapping = tkw.IndexMapping(
#        num_iterators=5, inputs={B: i, BN: j, H: k, N: l, M: m}, outputs={B: i, BN: j, H: k, N: l, M: m}
#    )
#    dv_mapping = tkw.IndexMapping(
#        num_iterators=5, inputs={B: i, BN: j, H: k, N: l, K: m}, outputs={B: i, BN: j, H: k, N: l, K: m}
#    )
#
#    @tkw.wave(constraints)
#    def batched_gemm_grad_v(
#        do: tkl.Memory[B, BN, M, H, K, GLOBAL_ADDRESS_SPACE, dtype],
#        a: tkl.Memory[B, BN, H, M, N, GLOBAL_ADDRESS_SPACE, dtype],
#        dv: tkl.Memory[B, BN, N, H, K, GLOBAL_ADDRESS_SPACE, dtype],
#    ):
#        dv_reg = tkl.Register[B, BN, H, N, K, tkl.f32](0.0)
#
#        @tkw.reduction(M, init_args=[dv_reg])
#        def repeat(
#            acc: tkl.Register[B, BN, H, N, K, tkl.f32]
#        ) -> tkl.Register[B, BN, H, N, K, tkl.f32]:
#            do_reg = tkw.read(do, mapping=do_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
#            if dtype == tkl.bf16:
#                do_reg = tkw.cast(tkw.cast(do_reg, tkl.f32), tkl.f16)
#            a_reg = tkw.read(a, mapping=a_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
#            if dtype == tkl.bf16:
#                a_reg = tkw.cast(tkw.cast(a_reg, tkl.f32), tkl.f16)
#            acc = tkw.mma(a_reg, do_reg, acc)
#            return acc
#
#        casted = tkw.cast(repeat, dtype)
#        tkw.write(casted, dv, mapping=dv_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)
#
#    hyperparams = {
#        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
#        LOAD_ELEMS_PER_THREAD: 4,
#        STORE_ELEMS_PER_THREAD: 4,
#        # WTF why tile sizes affect correctness?
#      # BLOCK_B: 1,
#      # BLOCK_BN: 1,
#      # BLOCK_M: 64,
#      # BLOCK_N: 64,
#      # BLOCK_H: 64,
#      # BLOCK_K: 32,
#
#        BLOCK_B: 1,
#        BLOCK_BN: 1,
#        BLOCK_M: 64,
#        BLOCK_N: 32,
#        BLOCK_H: 1,
#        BLOCK_K: 32,
#
#        B: batch,
#        BN: n,
#        M: q_seq_len,
#        N: kv_seq_len,
#        H: heads,
#        K: dim,
#    }
#    hyperparams.update(get_default_scheduling_params())
#    config = get_default_run_config()
#    if run_bench:
#        config["benchmark_batch_size"] = 10
#        config["benchmark_repetitions"] = 3
#    if dump_perf is not None:
#        perf_filename = request.node.name + ".json"
#        config["benchmark_results_file"] = os.path.join(
#            dump_perf, "tk_" + perf_filename
#        )
#
#    with tk.gen.TestLaunchContext(
#        hyperparams,
#        canonicalize=True,
#        run=True,
#        run_bench=run_bench,
#        run_config=config,
#        schedule=enable_scheduling,
#        use_scheduling_barriers=enable_scheduling_barriers,
#    ):
#        if dtype == tkl.bf16:
#            torch_dtype = torch.bfloat16
#        else:
#            torch_dtype = torch.float16
#
#        do = device_randn(batch, n, q_seq_len, heads, dim, dtype=torch_dtype)
#        a = device_randn(batch, n, heads, q_seq_len, kv_seq_len, dtype=torch_dtype)
#        dv = device_randn(batch, n, kv_seq_len, heads, dim, dtype=torch_dtype)
#        mb = batched_gemm_grad_v(do, a, dv)
#
#        if test_dump_generated_mlir:
#            filename = f"wave_batched_gemm_{'x'.join(map(str, shape))}.mlir"
#            with open(filename, "w") as f:
#                f.write(mb.module_op)
#
#        if run_bench:
#            if dump_perf is not None:
#                config["benchmark_results_file"] = os.path.join(
#                    dump_perf, "iree_" + perf_filename
#                )
#
#        torch_ref = torch.matmul(a.transpose(-1, -2), do.transpose(-2, -3)).transpose(-2, -3)
#        assert_close(dv, torch_ref, atol=3e-3, rtol=8e-3, check_dtype=False)
