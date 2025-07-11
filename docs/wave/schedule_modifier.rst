Schedule Validator
==================

The ScheduleValidator is a component that validates and optimizes operation schedules while maintaining resource and dependency constraints. It ensures that operations are scheduled in a way that respects both hardware resource limitations and data dependencies between operations.

Key Features
------------

- Validates and repairs schedules to maintain resource constraints
- Handles both forward and backward schedule repairs using a directional constraint enforcement strategy
- Supports modulo scheduling with a specified initiation interval (T)
- Tracks resource usage across scheduling cycles
- Maintains dependency relationships between operations

Components
----------

1. ResourceUsageTracker

   - Tracks resource usage across scheduling cycles
   - Validates if operations can be added at specific cycles
   - Maintains a global resource table (RT_global)

2. ScheduleDependencyGraph

   - Manages dependency relationships between operations
   - Provides methods to query predecessors and successors
   - Validates edge relationships

3. ScheduleConstraintRepairer

   - Repairs schedule violations by moving operations
   - Uses a directional constraint enforcement strategy:

     * Forward repair: Only enforces predecessor constraints
     * Backward repair: Only enforces successor constraints
   - Ensures both resource and dependency constraints are satisfied
   - Avoids unnecessary cascading repairs by processing nodes in order

Directional Constraint Enforcement
----------------------------------

The repair algorithm uses a directional strategy to efficiently maintain schedule validity:

Forward Repair
~~~~~~~~~~~~~~

When moving nodes forward in time:

- Nodes are processed in ascending order of their scheduled cycles
- Only predecessor constraints are enforced (node must start after its predecessors)
- Successor constraints are checked but not enforced
- This is because:

  * Successors will be processed later in the algorithm
  * If a successor needs to move, it will be handled when we process that node
  * Note: Moving a node forward can violate successor constraints, but we handle this by processing nodes in order

Backward Repair
~~~~~~~~~~~~~~~

When moving nodes backward in time:

- Nodes are processed in descending order of their scheduled cycles
- Only successor constraints are enforced (node must start before its successors)
- Predecessor constraints are checked but not enforced
- This is because:

  * Predecessors will be processed later in the algorithm
  * If a predecessor needs to move, it will be handled when we process that node
  * Note: Moving a node backward can violate predecessor constraints, but we handle this by processing nodes in order

This directional strategy makes the repair algorithm more efficient and predictable,
while still maintaining schedule validity. By processing nodes in order and only
enforcing constraints in the direction of movement, we avoid unnecessary cascading
repairs while ensuring a valid schedule. The key is that we process nodes in a
specific order (forward or backward) and handle any constraint violations when we
reach the affected nodes in that order.

Worked Example
--------------

Let's walk through an example using a simple graph with 4 operations (a, b, c, d) and 2 resource types:

.. code-block:: python

    # Create a graph with 4 operations
    graph = fx.Graph()
    target = lambda _: None

    # Create nodes with resource requirements
    a = graph.create_node("call_function", target, args=(), name="a")
    a.rrt = np.array([[1, 0]])  # Uses resource 0
    b = graph.create_node("call_function", target, args=(a,), name="b")
    b.rrt = np.array([[0, 1]])  # Uses resource 1
    c = graph.create_node("call_function", target, args=(b,), name="c")
    c.rrt = np.array([[1, 0]])  # Uses resource 0
    d = graph.create_node("call_function", target, args=(c,), name="d")
    d.rrt = np.array([[0, 1]])  # Uses resource 1

    # Create edges with latencies
    edges = [
        Edge(a, b, EdgeWeight(0, 2)),  # a->b with latency 2
        Edge(b, c, EdgeWeight(0, 1)),  # b->c with latency 1
        Edge(c, d, EdgeWeight(0, 1)),  # c->d with latency 1
    ]

Initial Schedule
~~~~~~~~~~~~~~~~

The initial schedule places operations with some spacing to allow for moves:

.. code-block:: python

    initial_schedule = {
        nodes["a"]: 0,  # Uses resource 0
        nodes["b"]: 3,  # Uses resource 1, a+2=2 but placed at 3
        nodes["c"]: 5,  # Uses resource 0, b+1=4 but placed at 5
        nodes["d"]: 6,  # Uses resource 1, c+1=6
    }

Creating the Validator
~~~~~~~~~~~~~~~~~~~~~~

We create a ScheduleValidator with:

- Modulo scheduling period (T) = 4
- Resource limits = [2, 2] (2 units each of resource 0 and 1)
- The graph nodes and edges defined above

.. code-block:: python

    validator = ScheduleValidator(
        initial_schedule=initial_schedule,
        T=4,
        nodes=list(nodes.values()),
        resource_limits=np.array([2, 2]),
        node_rrt_getter=lambda node: node.rrt,
        raw_edges_list=edges,
        num_resource_types=2,
    )

Attempting a Move
~~~~~~~~~~~~~~~~~

Let's try to move operation 'c' to cycle 2, which would require backward repair:

.. code-block:: python

    success, new_schedule, _ = validator.attempt_move(nodes["c"], 2)

The validator will:

1. Check if the move is possible
2. If possible, repair the schedule to maintain constraints:
   - For backward repair, only enforce successor constraints (c->d)
   - Don't enforce predecessor constraints (b->c) as they'll be handled when processing node 'b'
3. Return the new valid schedule

Result
~~~~~~

The move succeeds and produces a new schedule where:

- Operation 'c' is moved to cycle 2
- Operation 'b' is moved earlier to maintain the dependency (b->c)
- Operation 'a' remains at cycle 0
- Operation 'd' remains at cycle 6

The new schedule satisfies:

1. Resource constraints:

   - Operations 'a' and 'c' (using resource 0) don't overlap
   - Operations 'b' and 'd' (using resource 1) don't overlap

2. Dependency constraints:

   - a->b: b starts at least 2 cycles after a
   - b->c: c starts at least 1 cycle after b
   - c->d: d starts at least 1 cycle after c

Usage
-----

The ScheduleValidator is typically used in an optimization loop:

1. Start with an initial schedule
2. Attempt to move operations to improve the schedule
3. If a move succeeds, commit the new schedule
4. Repeat until no more improvements can be made

Example usage in an optimization loop:

.. code-block:: python

    # Try to move an operation
    success, new_schedule, error = validator.attempt_move(node, new_cycle)

    if success:
        # Get the new resource tracking state
        _, new_rt = validator.get_current_schedule_state()
        # Commit the move
        validator.commit_move(new_schedule, new_rt)
    else:
        # Handle the failed move
        print(f"Move failed: {error}")

The validator ensures that all moves maintain both resource and dependency constraints, making it a crucial component in schedule optimization.
