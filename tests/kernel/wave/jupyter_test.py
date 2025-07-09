# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel._support.indexing import sym
from iree.turbine.kernel._support.dtype import f16, f32
from iree.turbine.kernel.lang.wave_types import *
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import set_default_run_config
import iree.turbine.kernel as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from .common.utils import require_e2e
import torch

# Define symbolic dimensions for our matrices
M = sym.M  # Rows of A and C
N = sym.N  # Rows of B and columns of C
K = sym.K  # Columns of A and B

# Define workgroup tile sizes
BLOCK_M = sym.BLOCK_M
BLOCK_N = sym.BLOCK_N
BLOCK_K = sym.BLOCK_K

# Define the address space for our memory
ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

# Define constraints for the kernel
constraints = [
    tkw.WorkgroupConstraint(M, BLOCK_M, 0),
    tkw.WorkgroupConstraint(N, BLOCK_N, 1),
    tkw.TilingConstraint(K, BLOCK_K),
    tkw.WaveConstraint(M, BLOCK_M / 2),
    tkw.WaveConstraint(N, BLOCK_N / 2),
    tkw.HardwareConstraint(
        threads_per_wave=64,
        mma_type=tkw.MMAType.F32_16x16x16_F16,
    ),
]


@tkw.wave(constraints)
def gemm(
    a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
    b: Memory[N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
    c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
):
    # Initialize the accumulator register with zeros
    c_reg = Register[M, N, f32](0.0)

    # Iterate over the K dimension to compute the dot product
    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
        # Load elements from A and B
        a_reg = tkw.read(a)
        b_reg = tkw.read(b)

        # Compute matrix multiplication and accumulate
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    # Store the final result to C
    tkw.write(repeat, c)


@require_e2e
def test_gemm():
    # Create test matrices
    m, n, k = 128, 256, 128  # Small dimensions for testing

    # Initialize input matrices with random values
    torch.manual_seed(0)
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_B: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
    }

    # Compile the kernel
    options = WaveCompileOptions(subs=hyperparams, canonicalize=True)
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm)

    # Run the GEMM kernel
    compiled_gemm(a, b, c)

    # Verify the result using PyTorch's matmul
    expected = torch.matmul(a, b.t())

    # Check if results are close (accounting for floating-point precision)
    assert torch.allclose(
        c.to(torch.float16), expected, rtol=1e-2, atol=1e-2
    ), f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

    print("GEMM test passed!")
