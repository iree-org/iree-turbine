Schedule Reordering Module
==========================

This module implements a ping-pong scheduling algorithm for optimizing wave-level parallelism in GPU kernels. The algorithm enables two waves within the same SIMD group to run in parallel by carefully orchestrating their execution phases and synchronization.
Most of the module and algorithm described below can be found in ``iree-turbine/iree/turbine/kernel/wave/schedule_reordering.py``

Overview
--------

The scheduling algorithm implements a ping-pong pattern where two waves alternate between different computational phases:

1. Wave Synchronization Phase:

   * Wave "High" is initially blocked before the loop
   * Wave "Low" proceeds with memory operations
   * When Wave "Low" hits a workgroup barrier after reads, Wave "High" is unblocked
   * Both waves then proceed in parallel with different tasks

2. Parallel Execution Phases:

   First Cluster:

   * Wave "Low" performs matrix multiplications (MMAs)
   * Wave "High" performs local and global memory reads

   Second Cluster:

   * Wave "Low" performs local writes
   * Wave "High" performs matrix multiplications (MMAs)

3. Synchronization:

   * Waves synchronize using workgroup barriers
   * Conditional barriers control wave execution flow
   * After synchronization, both waves can proceed to the next iteration

Implementation Details
-----------------------

Operation Classification
~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm classifies operations into four main categories:

* Global Reads: Memory operations loading from global memory
* Local Reads: Memory operations loading from shared memory
* Local Writes: Memory operations writing to shared memory
* MMAs: Matrix multiplication operations

Graph Reordering
~~~~~~~~~~~~~~~~

The reordering process follows these steps:

1. Detection:

   * Identifies operations within a for-loop
   * Classifies operations into the four categories
   * Determines operation dependencies

2. Clustering:

   * Groups operations into logical clusters
   * Reorders clusters to optimize wave parallelism
   * Maintains data dependencies while enabling parallel execution

3. Graph Reconstruction:

   * Preserves operations before the cluster
   * Inserts reordered cluster operations
   * Maintains operations after the cluster
   * Ensures correct execution order while enabling parallel wave execution

Scheduling Strategy
-------------------

The module supports different scheduling strategies:

* ``SchedReorderStrategy.NONE``: No reordering applied
* ``SchedReorderStrategy.TWO_PP_CLUSTER``: Implements the ping-pong pattern with two clusters

The strategy selection is based on:

* Hardware constraints
* Tile sizes (M, N, K dimensions)
* Wave count requirements

Key Functions
-------------

.. function:: schedule_reordering(trace, constraints, scheduling_type)

   Main entry point for the scheduling algorithm. Processes the trace and applies
   the appropriate reordering strategy based on the given constraints.

.. function:: transform_two_PP_clusters(mma_nodes, local_load_lhs, local_load_rhs, global_load_lhs, global_load_rhs, local_write_lhs, local_write_rhs)

   Implements the two-cluster ping-pong transformation, creating the necessary
   operation clusters and synchronization points.

.. function:: add_conditional_barriers_to_loop(custom_iterate, trace, hardware_constraint)

   Adds conditional barriers to control wave execution flow, implementing the
   wave synchronization mechanism.

.. function:: reorder_graph(graph, clusters)

   Reconstructs the computation graph with the reordered operations while
   maintaining correct execution order.

Hardware Requirements
---------------------

The algorithm requires specific hardware characteristics:

* Even number of waves per block
* Compatible tile sizes for M, N, and K dimensions
* Support for wave-level synchronization primitives

The current implementation specifically targets configurations with 8 waves per block.

Example Configuration
---------------------

The default configuration for two-cluster ping-pong scheduling:

* Block M: 128
* Block N: 256
* Block K: 64
* Waves per block: 8

Notes
-----

* The algorithm is specifically designed for prefetch scheduling types
* Success of the transformation depends on the ability to properly classify
  and reorder operations
* The implementation includes safety checks to ensure correct execution
  order is maintained
* Wave synchronization is critical for correct execution and is handled
  through a combination of conditional and workgroup barriers
