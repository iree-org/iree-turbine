# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel.wave.iree_utils import generate_iree_ref
import os
import json

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))
require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")
# Whether to dump the generated MLIR module.
test_dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))

default_test_shapes = [(1024, 5120, 640), (2048, 10240, 1280), (4096, 20480, 2560)]

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
@pytest.mark.parametrize("shape", get_test_shapes("test_gemm"))
def testGemm(shape: tuple[int]):

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
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
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
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}
    with tk.gen.TestLaunchContext(
        hyperparams, canonicalize=True, run=True, run_config=config
    ):
        a = torch.randn(shape[0], shape[2], dtype=torch.float16)
        b = torch.randn(shape[1], shape[2], dtype=torch.float16)
        c = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        mb = gemm(a, b, c)

        if test_dump_generated_mlir:
            filename = f"wave_gemm_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        iree_ref = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        generate_iree_ref("mmt", [a, b], [iree_ref], config)
        assert torch.equal(c, iree_ref)


# Format: (M, K, N, B)
intermediate_size = 28672
tensor_parallel_shape = 8
hidden_size = 8192
# Batch size can be 1, 2, 3, 4
batch_size = 1
gemm_silu_shapes = [
    (
        intermediate_size / tensor_parallel_shape,
        hidden_size,
        hidden_size,
        1,
    )
]


@require_e2e
@pytest.mark.parametrize("shape", gemm_silu_shapes)
def testGemmSilu(shape: tuple[int]):

    # FC1 and FC2 GEMM Sizes
    # Weights matrices are of size (M0, K0).
    # Input matrix is of size (BS, K0).
    M = tkl.sym.M  # Reduction
    K = tkl.sym.K  # Reduction
    N = tkl.sym.N  # Parallel
    B = tkl.sym.B  # Parallel

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_B = tkl.sym.BLOCK_B

    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.TilingConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 2, 1))
    ]

    @tkw.wave(constraints)
    def gemm_silu(
        x: tkl.Memory[B, K, ADDRESS_SPACE, tkl.f16],
        w0: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        w1: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        w2: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
        output: tkl.Memory[B, N, ADDRESS_SPACE, tkl.f32],
    ):

        c_reg = tkl.Register[B, N, tkl.f32](0.0)

        @tkw.reduction(M, init_args=[c_reg])
        def outer_loop(acc: tkl.Register[B, N, tkl.f32]) -> tkl.Register[B, N, tkl.f32]:

            c_reg0 = tkl.Register[B, M, tkl.f32](0.0)
            c_reg1 = tkl.Register[B, M, tkl.f32](0.0)

            @tkw.reduction(K, init_args=[c_reg0, c_reg1])
            def inner_loop(
                acc0: tkl.Register[B, M, tkl.f32], acc1: tkl.Register[B, M, tkl.f32]
            ) -> tuple[tkl.Register[B, M, tkl.f32], tkl.Register[B, M, tkl.f32]]:
                x_reg = tkw.read(x, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                w0_reg = tkw.read(w0, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc0 = tkw.mma(x_reg, w0_reg, acc0)
                w1_reg = tkw.read(w1, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc1 = tkw.mma(x_reg, w1_reg, acc1)
                return acc0, acc1

            w2_reg = tkw.read(w2, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            mm0, mm1 = inner_loop
            silu = 1.0 / (1.0 + tkw.exp(-mm0))
            y = silu * mm1
            acc = tkw.mma(y, w2_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(outer_loop, output, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: 4,
        STORE_ELEMS_PER_THREAD: 4,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: shape[0],
        K: shape[1],
        N: shape[2],
        B: shape[3],
    }
    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}
    with tk.gen.TestLaunchContext(
        hyperparams, canonicalize=True, run=True, run_config=config
    ):
        x = torch.randn(shape[3], shape[1], dtype=torch.float16)
        w0 = torch.randn(shape[0], shape[1], dtype=torch.float16)
        w1 = torch.zeros(shape[0], shape[1], dtype=torch.float16)
        w2 = torch.zeros(shape[2], shape[0], dtype=torch.float16)
        output = torch.zeros(shape[3], shape[2], dtype=torch.float32)
        gemm_silu(x, w0, w1, w2, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
