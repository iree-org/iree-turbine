# TKW: A Wave Kernel Language for GPUs

TKW is a high-level programming language designed to simplify the development of GPU micro-kernels by abstracting over intricate details of GPU hardware. It allows developers to write efficient micro-kernels, focusing on core computations while inferring the required data transfers and indexing automatically. TKW implements a wave-based programming model to express programs leveraging coalesced memory accesses effortlessly and supports the explicit use of matrix multiplication intrinsics.

## Design Goals
TKW is designed with several key goals in mind to facilitate efficient GPU programming and maximize performance:

1. Abstract over hardware details: Simplify the development of GPU micro-kernels by hiding the complex details of synchronization, thread management, and memory transactions.
  - Automatically infer efficient data movement strategies across the memory hierarchy, ensuring efficient use of memory bandwidth.
  - Leverage hardware details (such as instruction specifications) to determine indexing.
2. Provide users with low-level control
  - Expose an interface to customize the instruction scheduling
  - Provide low-level control over how the computation is performed by exposing low-level GPU instructions. This empowers developers to directly leverage hardware-specific features to achieve maximum performance.
3. Systematically expressing constraints to leverage solvers / auto-tuning
  - Represent specific tiling possibilities around a micro-kernel using symbolic constraints. This forms a searchable space that allows for fine-tuning by exploring various tiling configurations.

## Wave-based Programming Model
TKW leverages a wave-based programming model that is specifically designed to take advantage of the parallel processing power of GPUs.

In GPU programming, a wavefront is a group of threads (or work items) that execute the same instruction in lockstep. In particular, coalesced memory accesses by all threads in a wavefront are executed together. This is analogous to the concept of a "warp" in NVIDIA's CUDA programming model.
Typically, on AMD GPUs, a wavefront contains 32 or 64 threads, all of which participate in executing the same instruction at the same time.

In this representation, memory access patterns are more naturally optimized for coalescing, reducing the complexity and manual effort required to achieve optimized memory transactions. In consequence, programmers can focus more on the core computational logic rather than the intricacies of thread coordination and memory management.
This approach contrasts with traditional models like OpenCL and CUDA, which often require more explicit management of threads and synchronization.

## Gemm example

https://github.com/iree-org/iree-turbine/blob/05652596420412c15c4b629df5c8748e98f5b8f2/tests/kernel/wave_gemm_test.py#L12-L71
