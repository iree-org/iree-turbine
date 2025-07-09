Global Load Minimization
========================

This document explains Wave's global load minimization optimization, which is a crucial pass in the compilation pipeline that reduces memory traffic by optimizing how data is loaded from global memory.

Overview
--------

The global load minimization pass transforms the kernel's memory access patterns to:

1. Reduce the number of global memory loads
2. Improve memory access coalescing

The pass achieves this by transforming the memory access pattern from an MMA (Matrix Multiply-Accumulate) access pattern to a linear access pattern. This transformation allows threads to load larger contiguous chunks of data from global memory, maximizing memory bandwidth utilization. Instead of having each thread load small, scattered elements needed for MMA operations, the pass reorganizes the access pattern so that threads can load larger, coalesced blocks of data that will be used across multiple MMA operations.

The following diagram illustrates the transformation of the computation graph:

.. mermaid::
   :caption: Computation Graph Transformation

   graph TB
       subgraph "Before Optimization"
           direction TB
           A1[Global Memory A] -->|"read(4)"| B1[Thread 0]
           A1 -->|"read(4)"| B2[Thread 1]
           A1 -->|"read(4)"| B3[Thread 2]
           A1 -->|"read(4)"| B4[Thread 3]

           C1[Global Memory B] -->|"read(4)"| B1
           C1 -->|"read(4)"| B2
           C1 -->|"read(4)"| B3
           C1 -->|"read(4)"| B4

           B1 -->|"write"| D1[Shared Memory A]
           B2 -->|"write"| D1
           B3 -->|"write"| D1
           B4 -->|"write"| D1

           B1 -->|"write"| D2[Shared Memory B]
           B2 -->|"write"| D2
           B3 -->|"write"| D2
           B4 -->|"write"| D2

           D1 -->|"read(4)"| E1[Thread 0]
           D1 -->|"read(4)"| E2[Thread 1]
           D1 -->|"read(4)"| E3[Thread 2]
           D1 -->|"read(4)"| E4[Thread 3]

           D2 -->|"read(4)"| E1
           D2 -->|"read(4)"| E2
           D2 -->|"read(4)"| E3
           D2 -->|"read(4)"| E4

           E1 -->|"mma"| F1[Accumulator]
           E2 -->|"mma"| F1
           E3 -->|"mma"| F1
           E4 -->|"mma"| F1

           F1 -->|"write"| G1[Output C]
       end

       subgraph "After Optimization"
           direction TB
           H1[Global Memory A] -->|"read(8)"| I1[Thread 0]
           H1 -->|"read(8)"| I2[Thread 1]

           J1[Global Memory B] -->|"read(8)"| I1
           J1 -->|"read(8)"| I2

           I1 -->|"write"| K1[Shared Memory A]
           I2 -->|"write"| K1

           I1 -->|"write"| K2[Shared Memory B]
           I2 -->|"write"| K2

           K1 -->|"barrier"| L1[Sync Point]
           K2 -->|"barrier"| L1

           L1 -->|"read(4)"| M1[Thread 0]
           L1 -->|"read(4)"| M2[Thread 1]
           L1 -->|"read(4)"| M3[Thread 2]
           L1 -->|"read(4)"| M4[Thread 3]

           K1 -->|"read(4)"| M1
           K1 -->|"read(4)"| M2
           K1 -->|"read(4)"| M3
           K1 -->|"read(4)"| M4

           K2 -->|"read(4)"| M1
           K2 -->|"read(4)"| M2
           K2 -->|"read(4)"| M3
           K2 -->|"read(4)"| M4

           M1 -->|"mma"| N1[Accumulator]
           M2 -->|"mma"| N1
           M3 -->|"mma"| N1
           M4 -->|"mma"| N1

           N1 -->|"write"| O1[Output C]
       end

       %% Styling
       classDef globalMem fill:#E6F3FF,stroke:#333,stroke-width:2px
       classDef sharedMem fill:#FFF9E6,stroke:#333,stroke-width:2px
       classDef thread fill:#CCCCCC,stroke:#333,stroke-width:2px
       classDef sync fill:#FFE6E6,stroke:#333,stroke-width:2px
       classDef acc fill:#E6FFE6,stroke:#333,stroke-width:2px
       classDef output fill:#FFE6FF,stroke:#333,stroke-width:2px

       class A1,C1,H1,J1 globalMem
       class D1,D2,K1,K2 sharedMem
       class B1,B2,B3,B4,E1,E2,E3,E4,I1,I2,M1,M2,M3,M4 thread
       class L1 sync
       class F1,N1 acc
       class G1,O1 output

The diagram above shows how the optimization transforms the computation graph:

1. **Before Optimization**:

   - Each thread performs small (4 elements) reads from global memory
   - Direct memory access to global memory for both input matrices

2. **After Optimization**:

   - Threads perform larger (8 elements) coalesced reads from global memory
   - Barrier synchronization point
   - Better memory access coalescing

How It Works
------------

The optimization process consists of several key steps:

1. **Analysis Phase**

   - Identifies global memory access patterns
   - Analyzes memory access dependencies
   - Determines potential for coalescing
   - Maps access patterns to thread indices

2. **Transformation Phase**

   - Coalesces global memory loads
   - Inserts appropriate barriers
   - Updates memory access indices


Example
-------

Let's look at a GEMM kernel that will be optimized by the global load minimization pass:

.. code-block:: python

    @tkw.wave(constraints)
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE_0, f16],  # Global memory
        b: Memory[N, K, ADDRESS_SPACE_0, f16],  # Global memory
        c: Memory[M, N, ADDRESS_SPACE, f32],    # Output
    ):
        c_reg = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Direct global memory loads
            a_reg = tkw.read(a, elements_per_thread=4)
            b_reg = tkw.read(b, elements_per_thread=4)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

This kernel performs matrix multiplication C = A @ B.T, where:
- A and B are f16 matrices in global memory
- C is an f32 output matrix
- Each thread loads 4 elements at a time from global memory
- The computation is performed using matrix multiply-accumulate (MMA) operations

The global load minimization pass will transform this kernel to:

1. Use larger, coalesced global memory loads
2. Add proper synchronization

Key Transformations
-------------------

1. **Memory Access Coalescing**

   - Combines multiple small loads into larger, aligned loads
   - Reduces memory transaction overhead
   - Improves memory bandwidth utilization
   - Transforms MMA access patterns to linear access patterns for better memory throughput

2. **Thread Synchronization**

   - Inserts barriers at appropriate points
   - Ensures correct memory access ordering
   - Maintains program correctness


Performance Impact
------------------

The global load minimization optimization typically provides:

1. **Reduced Memory Traffic**

   - Fewer global memory transactions
   - Better memory bandwidth utilization
   - Lower memory latency impact

2. **Improved Memory Access Patterns**

   - Coalesced global memory access


IR Transformation
-----------------

The optimization transforms the IR to optimize memory access patterns. Here's a simplified view of the key changes:

Before Optimization:
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    func.func @gemm(%a: !stream.binding<memref<MxKxf16>>,
                   %b: !stream.binding<memref<NxKxf16>>,
                   %c: !stream.binding<memref<MxNxf32>>) {
      %a_shared = wave.allocate((M, K), (BLOCK_M, BLOCK_K + 4), f16, shared)
      %b_shared = wave.allocate((N, K), (BLOCK_N, BLOCK_K + 4), f16, shared)

      %0 = wave.iterate(%K, [%c_reg], "region_0", [], 0, true)
      {
        %1 = wave.read(%a, 4) : memref<MxKxf16>
        wave.write(%1, %a_shared)
        %2 = wave.read(%b, 4) : memref<NxKxf16>
        wave.write(%2, %b_shared)

        %3 = wave.read(%a_shared, 4) : memref<MxKxf16>
        %4 = wave.read(%b_shared, 4) : memref<NxKxf16>
        %5 = wave.mma(%3, %4, %acc) : memref<MxNxf32>
      }
      wave.write(%0, %c)
      return
    }

After Optimization:
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    func.func @gemm(%a: !stream.binding<memref<MxKxf16>>,
                   %b: !stream.binding<memref<NxKxf16>>,
                   %c: !stream.binding<memref<MxNxf32>>) {
      %a_shared = wave.allocate((M, K), (BLOCK_M, BLOCK_K + 4), f16, shared)
      %b_shared = wave.allocate((N, K), (BLOCK_N, BLOCK_K + 4), f16, shared)

      %0 = wave.iterate(%K, [%c_reg], "region_0", [], 0, true)
      {
        wave.barrier()
        %1 = wave.read(%a, 8) : memref<MxKxf16>
        wave.write(%1, %a_shared)
        %2 = wave.read(%b, 8) : memref<NxKxf16>
        wave.write(%2, %b_shared)
        wave.barrier()

        %3 = wave.read(%a_shared, 4) : memref<MxKxf16>
        %4 = wave.read(%b_shared, 4) : memref<NxKxf16>
        %5 = wave.mma(%3, %4, %acc) : memref<MxNxf32>
      }
      wave.write(%0, %c)
      return
    }

For more details on the implementation, see the source code in `iree/turbine/kernel/wave/minimize_global_loads.py`.

Related Optimizations
---------------------

The global load minimization pass works in conjunction with other Wave optimizations:

1. **Memory Promotion**

   - Moves data to faster memory levels
   - Optimizes memory hierarchy usage
   - See :doc:`shared_memory` for details

2. **Scheduling**

   - Optimizes operation ordering
   - Improves resource utilization
   - See :doc:`system_architecture` for details
