Schedule Optimization with tune_attention
=========================================

This document explains how the `tune_attention` function works to optimize attention kernel schedules using hill climbing optimization.

Overview
--------

The `tune_attention` function implements an automated schedule optimization process for vanilla attention kernels. It uses a hill climbing algorithm to iteratively improve the schedule by moving operations to different cycles while maintaining resource and dependency constraints.

Key Components
--------------

1. **ScheduleValidator**: Validates and modifies schedules while maintaining constraints
2. **ScheduleOptimizer**: Implements the hill climbing optimization algorithm
3. **Resource Reservation Table (RRT)**: Tracks resource usage across scheduling cycles
4. **TuningLogger**: Logs optimization progress and saves intermediate schedules

How It Works
------------

The optimization process follows these steps:

1. **Initial Schedule Generation**: Compile the kernel and extract the initial schedule
2. **Schedule Loading**: Load the schedule with resource requirements and dependencies
3. **Optimization Loop**: Iteratively improve the schedule using hill climbing
4. **Schedule Validation**: Ensure each move maintains resource and dependency constraints
5. **Performance Measurement**: Measure latency for each candidate schedule
6. **Progress Logging**: Save schedules and track improvements

Example Usage
------------

Here's a complete example of how to use `tune_attention`:

.. code-block:: python

    from iree.turbine.kernel.wave.tuner.tune_attention import (
        tune_attention, AttentionConfig
    )
    from iree.turbine.kernel.wave.constraints import MMAType
    from iree.turbine.kernel.wave.scheduling import SchedulingType

    # Configure the attention kernel
    config = AttentionConfig(
        batch_size=1,
        num_heads=32,
        seq_len_q=512,
        seq_len_k=512,
        head_dim=64,
        head_dim_kv=64,
        mfma_variant=(MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
        enable_scheduling=SchedulingType.MODULO,
        dynamic_dims=False,
        num_warmup=10,
        num_iterations=100,
    )

    # Run optimization
    best_schedule, best_latency = tune_attention(config)
    print(f"Best latency: {best_latency:.6f} seconds")

Configuration Parameters
------------------------

The `AttentionConfig` class controls the optimization process:

- **batch_size**: Number of batches to process
- **num_heads**: Number of attention heads
- **seq_len_q**: Query sequence length
- **seq_len_k**: Key sequence length
- **head_dim**: Dimension of each attention head
- **head_dim_kv**: Dimension of key/value heads
- **mfma_variant**: Matrix multiply-accumulate variant to use
- **enable_scheduling**: Type of scheduling to enable
- **dynamic_dims**: Whether dimensions are dynamic
- **num_warmup**: Number of warmup iterations for timing
- **num_iterations**: Number of benchmark iterations

Optimization Algorithm
---------------------

The hill climbing algorithm works as follows:

1. **Random Move Selection**: Randomly select a node and a new target cycle
2. **Move Validation**: Check if the move maintains all constraints
3. **Schedule Repair**: If needed, repair the schedule to satisfy dependencies
4. **Performance Measurement**: Measure the latency of the new schedule
5. **Improvement Check**: Accept the move if it improves performance
6. **Iteration**: Repeat until no improvement is found or max iterations reached

Schedule Validation
------------------

Each schedule modification is validated using the `ScheduleValidator`:

.. code-block:: python

    # Example of schedule validation
    validator = ScheduleValidator(
        initial_schedule=schedule,
        T=initiation_interval,
        nodes=nodes,
        resource_limits=resource_limits,
        node_rrt_getter=node_rrt_getter,
        raw_edges_list=edges,
        num_resource_types=num_resource_types,
    )

    # Attempt to move a node
    success, new_schedule, error_msg = validator.attempt_move(node, new_cycle)
    if success:
        validator.commit_move(new_schedule, new_rrt)

Resource Reservation Table (RRT)
-------------------------------

The RRT tracks resource usage across scheduling cycles:

.. code-block:: python

    # RRT structure: (num_cycles, num_resource_types)
    rrt = np.zeros((initiation_interval, num_resource_types), dtype=int)

    # Example RRT for a 4-cycle schedule with 3 resource types
    # Cycle | Resource1 | Resource2 | Resource3
    #   0   |     1     |     0     |     2
    #   1   |     0     |     2     |     1
    #   2   |     2     |     1     |     0
    #   3   |     1     |     1     |     1

The RRT is updated whenever nodes are moved to ensure resource constraints are maintained.

Output and Logging
-----------------

The optimization process generates several output files:

1. **Schedule Files**: Each iteration's schedule saved as both JSON and text
2. **Progress Log**: CSV file tracking latency improvements
3. **Tuning History**: JSON file with complete optimization history
4. **Final Results**: Summary of the best schedule found

Example output structure:

.. code-block:: text

    attention_tuning/
    └── tune_20250703_123456/
        ├── schedules/
        │   ├── schedule_0000.txt    # Initial schedule
        │   ├── schedule_0001.txt    # First improvement
        │   ├── schedule_0002.txt    # Second improvement
        │   └── ...
        ├── traces/
        │   ├── trace_0000.rpd       # RPD traces for timing
        │   └── ...
        ├── tuning.log               # Detailed log
        ├── tuning_progress.csv      # Progress tracking
        ├── tuning_history.json      # Complete history
        └── final_results.json       # Final summary

Performance Measurement
----------------------

Performance is measured using RPD (https://github.com/ROCm/rocmProfileData) when available:

.. code-block:: python

    def measure_with_rpd(kernel_fn, *args, num_warmup, num_iterations,
                        output_filename, config):
        # Warmup runs
        for _ in range(num_warmup):
            _ = kernel_fn(*args)

        # Benchmark runs with profiling
        tracer = rpdTracerControl()
        tracer.start()
        for _ in range(num_iterations):
            _ = kernel_fn(*args)
        tracer.stop()

        # Calculate average latency
        avg_time = calculate_average_latency(output_filename)
        return avg_time


Constraints and Validation
-------------------------

The optimization process maintains several constraints:

1. **Resource Constraints**: No cycle can exceed resource limits
2. **Dependency Constraints**: Predecessors must execute before successors
3. **Schedule Validity**: All nodes must be scheduled within valid cycles

Troubleshooting
--------------

Common issues and solutions:

1. **Compilation Failures**: Invalid schedules may cause compilation to fail

   - The system returns infinity latency for failed compilations
   - These schedules are automatically rejected

2. **No Improvements**: If no improvements are found

   - Check if the initial schedule is already optimal
   - Increase max_iterations or max_no_improvement parameters
   - Verify resource constraints are not too restrictive

3. **Resource Violations**: If resource constraints are violated

   - Check the RRT in schedule files
   - Verify resource limits are appropriate for the kernel

4. **Dependency Violations**: If dependency constraints are violated

   - Check the schedule file for cycles where predecessors > successors
   - Verify the dependency graph is correctly constructed

Advanced Usage
-------------

For advanced users, you can customize the optimization process:

.. code-block:: python

    # Custom optimization parameters
    result = optimizer.optimize(
        max_iterations=50,           # More iterations
        max_no_improvement=30,       # Longer patience
        verbose=True                 # Detailed logging
    )

    # Access optimization history
    print(f"Improvement history: {result.improvement_history}")
    print(f"Total iterations: {result.iterations}")

    # Save custom schedule files
    dump_schedule(
        schedule=result.schedule,
        initiation_interval=initiation_interval,
        num_stages=num_stages,
        dump_file="custom_schedule.txt",
        resource_reservations=resource_reservations,
        resource_names=resource_names
    )
