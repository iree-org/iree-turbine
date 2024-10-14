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
from iree.turbine.kernel.wave.packaging.build_package import create_pip_package
import os
import json
from torch.testing import assert_close


def packageTest():
    shape = (2048, 1280, 1280)
    enable_scheduling = True
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
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }
    config = {
        "backend": "rocm",
        "device": "hip",
        "target": "gfx942",
        "dump_vmfb_file": "artifacts/kernel.vmfb",
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=False,
        run_config=config,
        schedule=enable_scheduling,
    ):
        a = torch.randn(shape[0], shape[2], dtype=torch.float16)
        b = torch.randn(shape[1], shape[2], dtype=torch.float16)
        c = torch.zeros(shape[0], shape[1], dtype=torch.float32)
        gemm(a, b, c)

        # Create the pip package
        kernel_info = {
            "package_name": "libtkw",
            "kernel_name": "gemm_f32_2048x1280x1280_f16",
            "num_inputs": 2,
            "dispatch_name": "isolated_benchmark",
            "vmfb_path": "artifacts/kernel.vmfb",
            "kernel_version": "0.0.1",
        }
        create_pip_package(output_dir="pip_package/", kernel_info=kernel_info)
        # Run python setup.py bdist_wheel in pip_package/ to build the wheel.
        # Once the wheel is built, it can be installed using
        # pip install <wheel_name>.whl --find-links https://iree.dev/pip-release-links.html
        # The kernel can then be invoked from Python as follows:
        # import libtkw
        # import torch
        # a = torch.randn(2048, 1280, dtype=torch.float16, device="cuda")
        # b = torch.randn(1280, 1280, dtype=torch.float16, device="cuda")
        # c = torch.empty(2048, 1280, dtype=torch.float32, device="cuda")
        # libtkw.gemm_f32_2048x1280x1280_f16(a, b, c)
        assert os.path.exists("pip_package/libtkw-0.0.1-py3-none-any.whl")


packageTest()
