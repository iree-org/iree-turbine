Schedule File Format
====================

The schedule file format is used to store and load scheduling information for wave kernels. It provides a human-readable representation of the schedule, including metadata, resource usage, and operation timing.

File Structure
--------------

The schedule file consists of three main sections:

1. Metadata (required)
2. Resource Reservation Table (RRT) (optional)
3. Schedule Table (required)

Metadata Section
----------------

The metadata section appears at the top of the file and contains two required pieces of information:

.. code-block:: text

    Initiation Interval: <II>
    Number of Stages: <num_stages>

Where:
- ``<II>`` is the initiation interval (number of cycles in the repeating pattern)
- ``<num_stages>`` is the number of pipeline stages

Resource Reservation Table (RRT)
--------------------------------

The RRT section is optional and provides information about resource usage across the initiation interval. It is formatted as a table with the following structure:

.. code-block:: text

    # Resource Reservation Table (RRT):
    # Each row represents a cycle in the initiation interval
    # Each column represents a resource type
    # Format: cycle | resource_usage

    Cycle | GLOBAL_MEMORY_UNITS | SHARED_MEMORY_UNITS | MMA_UNITS | VALU_UNITS | SHUFFLE_UNITS
    ----- | ------------------- | ------------------- | --------- | ---------- | -------------
        0 |                   2 |                   4 |         4 |          0 |             0
        1 |                   0 |                   4 |         0 |          0 |             0
        2 |                   0 |                   2 |         4 |          0 |             0

The RRT shows how many resources of each type are used in each cycle of the initiation interval. This helps in understanding resource utilization and potential bottlenecks.

Schedule Table
--------------

The schedule table provides detailed information about each operation in the schedule. It is formatted as a pipe-delimited table with the following columns:

1. Node Name: The name of the operation
2. Node Type: The type of operation (e.g., ReadShared, WriteShared, MMA)
3. Node Sort Key: A unique identifier for the node
4. Cycle: The absolute cycle number when the operation is scheduled
5. Relative Cycle: The cycle within the initiation interval (cycle % II)
6. Stage: The pipeline stage (cycle // II)
7. User Sort Keys: The sort keys of nodes that depend on this operation

Example
-------

Here's a complete example of a schedule file:

.. code-block:: text

    Initiation Interval: 3
    Number of Stages: 3

    # Resource Reservation Table (RRT):
    # Each row represents a cycle in the initiation interval
    # Each column represents a resource type
    # Format: cycle | resource_usage

    Cycle | GLOBAL_MEMORY_UNITS | SHARED_MEMORY_UNITS | MMA_UNITS | VALU_UNITS | SHUFFLE_UNITS
    ----- | ------------------- | ------------------- | --------- | ---------- | -------------
        0 |                   2 |                   4 |         4 |          0 |             0
        1 |                   0 |                   4 |         0 |          0 |             0
        2 |                   0 |                   2 |         4 |          0 |             0

    Node Name                   | Node Type     | Node Sort Key   | Cycle   | Relative Cycle   | Stage   | User Sort Keys
    --------------------------- | ------------- | --------------- | ------- | ---------------- | ------- | ----------------------------
    read_21                     | ReadGlobal    | (4,)            | 0       | 0                | 0       | (5,)
    read_22                     | ReadGlobal    | (6,)            | 0       | 0                | 0       | (7,)
    write_10                    | WriteShared   | (5,)            | 2       | 2                | 0       | (12,), (13,), (14,), (15,)
    write_11                    | WriteShared   | (7,)            | 2       | 2                | 0       | (8,), (9,), (10,), (11,)
    --------------------------- | ------------- | --------------- | ------- | ---------------- | ------- | ----------------------------
    read_2_shared_M_1_N_0_K_1   | ReadShared    | (15,)           | 3       | 0                | 1       | (21,), (23,)
    read_2_shared_M_1_N_0_K_0   | ReadShared    | (14,)           | 3       | 0                | 1       | (20,), (22,)
    read_2_shared_M_0_N_0_K_1   | ReadShared    | (13,)           | 3       | 0                | 1       | (17,), (19,)
    read_4_shared_M_0_N_0_K_1   | ReadShared    | (9,)            | 3       | 0                | 1       | (17,), (21,)
    read_2_shared_M_0_N_0_K_0   | ReadShared    | (12,)           | 4       | 1                | 1       | (16,), (18,)
    read_4_shared_M_0_N_0_K_0   | ReadShared    | (8,)            | 4       | 1                | 1       | (16,), (20,)
    read_4_shared_M_0_N_1_K_0   | ReadShared    | (10,)           | 4       | 1                | 1       | (18,), (22,)
    read_4_shared_M_0_N_1_K_1   | ReadShared    | (11,)           | 4       | 1                | 1       | (19,), (23,)
    mma_M_0_N_0_K_0             | MMA           | (16,)           | 5       | 2                | 1       | (17,)
    mma_M_1_N_0_K_0             | MMA           | (20,)           | 5       | 2                | 1       | (21,)
    mma_M_0_N_1_K_0             | MMA           | (18,)           | 5       | 2                | 1       | (19,)
    mma_M_1_N_1_K_0             | MMA           | (22,)           | 5       | 2                | 1       | (23,)
    --------------------------- | ------------- | --------------- | ------- | ---------------- | ------- | ----------------------------
    mma_M_0_N_0_K_1             | MMA           | (17,)           | 6       | 0                | 2       | (0,)
    mma_M_1_N_0_K_1             | MMA           | (21,)           | 6       | 0                | 2       | (2,)
    mma_M_0_N_1_K_1             | MMA           | (19,)           | 6       | 0                | 2       | (1,)
    mma_M_1_N_1_K_1             | MMA           | (23,)           | 6       | 0                | 2       | (3,)
    acc_M_0_N_0_K_0             | IterArg       | (0,)            | 7       | 1                | 2       | (16,)
    acc_M_1_N_0_K_0             | IterArg       | (2,)            | 7       | 1                | 2       | (20,)
    acc_M_0_N_1_K_0             | IterArg       | (1,)            | 7       | 1                | 2       | (18,)
    acc_M_1_N_1_K_0             | IterArg       | (3,)            | 7       | 1                | 2       | (22,)

In this example:

- The initiation interval is 3 cycles
- There are 3 pipeline stages
- The RRT shows resource usage for each cycle in the initiation interval
- The schedule table shows all operations, with:
  - Operations in stage 0 (cycles 0-2)
  - Operations in stage 1 (cycles 3-5)
  - Operations in stage 2 (cycles 6-7)
- Separator lines (dashes) are used to visually separate different stages
- Each operation's dependencies are listed in the User Sort Keys column

Using Schedule Files
--------------------

Schedule files can be used in two ways:

1. Exporting a schedule: Use ``dump_schedule`` to save a computed schedule to a file
2. Loading a schedule: Use ``override_schedule`` to load a previously computed schedule

Example usage:

.. code-block:: python

    # Export a schedule
    options = WaveOptions(
        ...,
        dump_schedule="./schedule.txt",
    )

    # Load a schedule
    options = WaveOptions(
        ...,
        override_schedule="./schedule.txt",
    )
