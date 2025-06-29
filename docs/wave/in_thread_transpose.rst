In-Thread Transpose Optimization
================================

This document explains Wave's in-thread transpose optimization, which is a important pass in the compilation pipeline that optimizes transpose operations by performing them using contiguous vector ops rather than using gather/scatter.
Contiguous vector ops generally are more efficient than gathers/scatters.

Overview
--------

The in-thread transpose optimization detects when a read-write pair is performing a transpose operation and combines it with the global load minimization optimization. Instead of performing transpose using gather on global/shared memory like generic minimize global load pass does, we can instead load contiguous vectors from global memory and perform the transpose in each thread's registers using a sequence of vector extract and reshape operations.

The following diagram illustrates the transformation:

.. mermaid::
   :caption: In-Thread Transpose Transformation

   graph TB
       subgraph "in-thread Transpose"
           direction TB
           F1[Global Memory] -->|"read(8x4)"| G1[Thread 0 Registers]
           F1 -->|"read(8x4)"| G2[Thread 1 Registers]

           G1 -->|"transpose"| H1[Thread 0 Registers]
           G2 -->|"transpose"| H2[Thread 1 Registers]

           H1 -->|"write(4x8)"| I1[Shared Memory]
           H2 -->|"write(4x8)"| I1

           I1 -->|"read"| J1[Thread 0 Registers]
           I1 -->|"read"| J2[Thread 1 Registers]

           J1 -->|" "| K1[MMA]
           J2 -->|" "| K1
       end

       subgraph "Before Optimization"
           direction TB
           A1[Global Memory] -->|"gather read(4x8)"| B1[Thread 0 Registers]
           A1 -->|"gather read(4x8)"| B2[Thread 1 Registers]

           B1 -->|"write(4x8)"| C1[Shared Memory]
           B2 -->|"write(4x8)"| C1

           C1 -->|"read"| D1[Thread 0 Registers]
           C1 -->|"read"| D2[Thread 1 Registers]

           D1 -->|" "| E1[MMA]
           D2 -->|" "| E1
       end

       %% Styling
       classDef globalMem fill:#E6F3FF,stroke:#333,stroke-width:2px
       classDef sharedMem fill:#FFF9E6,stroke:#333,stroke-width:2px
       classDef thread fill:#CCCCCC,stroke:#333,stroke-width:2px
       classDef registers fill:#E6FFE6,stroke:#333,stroke-width:2px
       classDef output fill:#FFE6FF,stroke:#333,stroke-width:2px

       class A1,F1 globalMem
       class C1,I1 sharedMem
       class B1,B2,D1,D2,G1,G2,J1,J2,H1,H2 registers
       class E1,K1 output

How It Works
------------

The in-thread transpose optimization works through several key phases:

1. **Detection Phase**

   - Identifies read-write pairs
   - Analyzes index/mapping to determine if transpose is occurring
   - Validates that the optimization conditions are met

2. **Analysis Phase**

   - Determines the vector sizes for loading and storing
   - Calculates the number of loads and stores needed

3. **Transformation Phase**

   - Computes new memory access indices and mappings
   - Generates new read/write operations with new vector sizes
   - Creates vector transpose operations using extract/reshape sequences

Example
-------

Consider a GEMM kernel that performs matrix multiplication with a transposed input:

.. code-block:: python

    @tkw.wave(constraints)
    def gemm(
        a: Memory[M, K, ADDRESS_SPACE, f16],
        b: Memory[K, N, ADDRESS_SPACE, f16],  # Note: K, N layout
        c: Memory[M, N, ADDRESS_SPACE, f32],
    ):
        c_reg = Register[M, N, f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Read A in [M, K] layout
            a_reg = tkw.read(a)

            # Read B with mapping to transpose from [K, N] to [N, K]
            b_mapping = IndexMapping(
                num_iterators=2,
                inputs={N: iterator(0), K: iterator(1)},
                outputs={N: iterator(0), K: iterator(1)}
            )
            b_reg = tkw.read(b, mapping=b_mapping)

            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c)

In this example, the read of matrix B involves a transpose operation from [K, N] to [N, K] layout. The in-thread transpose optimization will transform the read of matrix B into a sequence of vector loads from global memory and a sequence of vector extract and reshape operations to perform the transpose in each thread's registers.

Limitations
-----------

The optimization has several limitations:

1. **Dimension Constraints**: Only supports transposing the last two dimensions
2. **Vector Size Requirements**: Requires compatible vector sizes for the hardware, vector ops count for the global read must be greater than 1
3. **No indirect indexing**: Indirect indexing is too hard to analyze yet
4. **No bank conflicts reduction**: It may be possible to reduce bank conflicts by using more complex indexing pattern, but it's not implemented yet
