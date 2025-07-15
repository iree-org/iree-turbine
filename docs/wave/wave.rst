Wave: High-Performance Machine Learning Programming Language
=============================================================================

Motivation
-----------

Wave is a high-level programming language designed to accelerate the development and optimization of machine learning kernels. it aims to dramatically improve kernel author velocity in two critical dimensions:

1. **Implementation Velocity**: Wave enables rapid prototyping of new optimization ideas and algorithms through its high-level abstractions and symbolic programming model. Kernel authors can quickly express complex tensor operations and experiment with different optimization strategies without getting bogged down in low-level implementation details.

2. **Performance Velocity**: The language is designed to achieve high performance with minimal tuning effort. Through its declarative constraints and hardware-aware abstractions, Wave automatically generates optimized GPU code, allowing kernel authors to focus on algorithmic innovation rather than manual optimization.

Wave is particularly focused on the machine learning domain, with deep integration into the PyTorch ecosystem. This integration enables:

- Seamless transition from PyTorch models to optimized GPU kernels
- Easy experimentation with new ML algorithms and optimizations
- Rapid deployment of custom kernels in production ML pipelines
- Direct reuse of PyTorch's tensor abstractions and operator semantics

The language bridges the gap between high-level ML programming and low-level GPU performance by providing a flexible and expressive programming model that maintains close control over hardware resources while enabling rapid innovation in the ML space.

Core Design Principles
-----------------------

Wave is built around several key design principles that guide its architecture and implementation:

1. **Symbolic Programming Model**

   - Heavy use of symbolic variables to represent tensor dimensions
   - Symbolic expressions for memory access patterns and layouts
   - Enables compile-time optimization and analysis
   - Provides flexibility in expressing complex tensor operations

2. **Separation of Distribution and Computation**

   - Clear separation between distribution strategy and computation graph
   - Distribution strategy defined through declarative constraints
   - Computation graph expressed independently of parallelization
   - Enables better optimization and code reuse

3. **Dual Programming Models**

   - Support for both tile-based and SIMT-based programming models
   - Tile-based model for coarse-grained parallelism
   - SIMT model for fine-grained vector operations
   - Flexible mapping to different GPU architectures

4. **Hardware-Aware Abstractions**

   - Direct mapping to GPU hardware concepts (workgroups, waves, etc.)
   - Explicit control over memory hierarchy
   - Hardware-specific optimizations (MMA operations, memory layouts)
   - Performance portability across different GPU architectures


For more detailed information about Wave's architecture and optimization passes, see :doc:`system_architecture`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   system_architecture
   shared_memory
   runtime
   gemm_tutorial
   minimize_global_loads
   pingpong_reordering
   schedule
   schedule_modifier
   fused_softmax
   aplp
