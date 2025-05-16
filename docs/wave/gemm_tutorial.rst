GEMM Tutorial
============

This tutorial demonstrates how to implement a high-performance matrix multiplication (GEMM) kernel using Wave. We'll walk through the implementation step by step, explaining the key concepts and optimizations.

Overview
--------

The GEMM kernel we'll implement computes C = A @ B.T, where:

- A is an M×K matrix in f16
- B is an N×K matrix in f16
- C is an M×N matrix in f32

We'll use Wave's symbolic programming model and hardware-aware abstractions to create an efficient implementation.

Implementation
-------------

First, we need to import the necessary modules and define our symbolic dimensions:

.. code-block:: python

    from iree.turbine.kernel._support.indexing import sym
    from iree.turbine.kernel._support.dtype import f16, f32
    from iree.turbine.kernel.lang.wave_types import *
    from iree.turbine.kernel.lang.global_symbols import *
    from iree.turbine.kernel.wave.utils.run_utils import set_default_run_config
    import iree.turbine.kernel as tkl
    import iree.turbine.kernel.wave as tkw
    from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
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
    ADDRESS_SPACE = sym.ADDRESS_SPACE
    GLOBAL_ADDRESS_SPACE = sym.GLOBAL_ADDRESS_SPACE

Now, let's define our GEMM kernel with appropriate constraints:

.. code-block:: python

    # Define constraints for the kernel
    constraints = [
        # Distribute M dimension across workgroups with tile size BLOCK_M
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        # Distribute N dimension across workgroups with tile size BLOCK_N
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        # Tile the K dimension for reduction
        tkw.TilingConstraint(K, BLOCK_K),
        # Further distribute M among waves with a tile size of BLOCK_M / 2
        tkw.WaveConstraint(M, BLOCK_M / 2),
        # Further distribute N among waves with a tile size of BLOCK_N / 2
        tkw.WaveConstraint(N, BLOCK_N / 2),
        # Hardware-specific constraints
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],  # Input matrix A
        b: Memory[N, K, ADDRESS_SPACE, f16],  # Input matrix B
        c: Memory[M, N, GLOBAL_ADDRESS_SPACE, f32],  # Output matrix C
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

Testing the Implementation
------------------------

Let's create a test function to verify our GEMM implementation:

.. code-block:: python

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
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            M: m,
            N: n,
            K: k,
        }

        # Compile the kernel
        options = WaveCompileOptions(
            subs=hyperparams,
        )
        options = set_default_run_config(options)
        compiled_gemm = wave_compile(options, gemm)

        # Run the GEMM kernel
        compiled_gemm(a, b, c)

        # Verify the result using PyTorch's matmul
        expected = torch.matmul(a, b.t())

        # Check if results are close (accounting for floating-point precision)
        assert torch.allclose(c.to(torch.float16), expected, rtol=1e-2, atol=1e-2), \
            f"GEMM result doesn't match expected output\nMax difference: {(c - expected).abs().max()}"

        print("GEMM test passed!")

Key Components
-------------

1. **Memory Types and Data Types**:

   - ``Memory[M, K, ADDRESS_SPACE, f16]`` defines a matrix in memory with dimensions M×K
   - ``f16`` and ``f32`` specify half and single precision floating-point types
   - Different address spaces (shared and global) for optimal memory access

2. **Wave Language Features**:

   - ``@tkw.wave()`` decorator with constraints defines the kernel's execution parameters
   - ``@tkw.iterate`` creates an iteration loop over the K dimension
   - ``Register`` represents values in registers during computation
   - ``tkw.read`` and ``tkw.write`` handle memory operations
   - ``tkw.mma`` performs matrix multiply-accumulate operations

3. **Constraints**:

   - **Workgroup Constraints**: Distribute computation across workgroups
     - M dimension is distributed with tile size BLOCK_M
     - N dimension is distributed with tile size BLOCK_N
   - **Wave Constraints**: Enable wave-level parallelism
     - M and N dimensions are further parallelized within workgroups
   - **Hardware Constraints**: Specify GPU-specific parameters
     - 64 threads per wave
     - 2x2x1 waves per block
     - F32_16x16x16_F16 matrix multiply-accumulate operation

4. **Memory Hierarchy**:

   - Input matrices (a, b) are in shared memory for fast access
   - Output matrix (c) is in global memory
   - Intermediate results are kept in registers

5. **Computation Flow**:

   - Initialize accumulator register with zeros
   - Iterate over K dimension to perform reduction
   - Load tiles from shared memory
   - Perform matrix multiplication and accumulation
   - Write final result to global memory

Performance Considerations
------------------------

1. **Tile Size Selection**:

   - Choose tile sizes that maximize memory locality
   - Consider hardware constraints (shared memory size, register file size)
   - Balance between parallelism and resource usage
   - Example values: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32

2. **Memory Access Patterns**:

   - Use shared memory for frequently accessed data (input matrices)
   - Minimize bank conflicts in shared memory
   - Align memory accesses for better coalescing
   - Consider mixed precision (f16 inputs, f32 accumulation)

3. **Wave Organization**:

   - Distribute work evenly across waves
   - Use hardware-specific wave sizes (64 threads per wave)
   - Optimize for the target GPU architecture
   - Consider wave-level parallelism for both M and N dimensions

4. **Testing and Validation**:

   - Use small test cases for initial verification
   - Compare against PyTorch's implementation
   - Account for floating-point precision differences
   - Use appropriate error tolerances (rtol=1e-2, atol=1e-2)

For more advanced optimizations and techniques, see the :doc:`system_architecture` documentation.
