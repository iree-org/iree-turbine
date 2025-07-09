Optimizing Shared Memory Allocations
=============================================================

This document explains the First-Fit strategy used to optimize shared memory allocations in the compiler.

The goal is to assign a starting memory offset to each allocation such that the total peak memory usage is minimized, while ensuring that no two allocations active at the same time occupy overlapping memory regions.

The heuristic provides a fast, approximate solution to this problem. It does not guarantee optimality but often performs well in practice.

The Allocation Data
--------------------

We will use the following set of allocations for this explanation:

.. code-block:: python

   allocations_data = [
       (10, 1, 5),  # Allocation 0: size 10, active from time 1 to 5
       (5, 2, 6),   # Allocation 1: size 5, active from time 2 to 6
       (8, 1, 3),   # Allocation 2: size 8, active from time 1 to 3
       (4, 4, 7),   # Allocation 3: size 4, active from time 4 to 7
       (6, 3, 8),   # Allocation 4: size 6, active from time 3 to 8
       (12, 5, 9),  # Allocation 5: size 12, active from time 5 to 9
   ]

Each tuple represents an allocation with ``(size, start_time, end_time)``. The times are discrete steps, and an allocation is active from ``start_time`` to ``end_time`` inclusive.

Heuristic Steps
---------------

The heuristic proceeds as follows:

1.  **Sort Allocations:** The allocations are first sorted in **decreasing order of their size**. This is the "Decreasing by Size" part of the heuristic. The intuition is that placing larger items first helps to potentially fill up space more efficiently.

    For our ``allocations_data``, the sorted order is:

    - Allocation 5: size 12, lifetime [5, 9]
    - Allocation 0: size 10, lifetime [1, 5]
    - Allocation 2: size 8, lifetime [1, 3]
    - Allocation 4: size 6, lifetime [3, 8]
    - Allocation 1: size 5, lifetime [2, 6]
    - Allocation 3: size 4, lifetime [4, 7]

2.  **Iterate and Place:** The heuristic then iterates through the allocations in this sorted order. For each allocation, it finds the first available memory offset where it can be placed without conflicting with any *already placed* allocation whose lifetime overlaps. This is the "First-Fit" part.

3.  **Finding the First Available Offset (First-Fit with Optimization):** For the current allocation being placed, the heuristic checks potential starting offsets, typically starting from 0 and increasing. For a given ``current_offset``, it checks for conflicts with every allocation that has already been placed.

    A conflict exists between the current allocation (with size :math:`S_{current}`, lifetime :math:`[T_{start\_current}, T_{end\_current}]`) at ``current_offset`` and a placed allocation (with size :math:`S_{placed}`, lifetime :math:`[T_{start\_placed}, T_{end\_placed}]`, offset :math:`O_{placed}`) if **both** of the following conditions are true:

    a.  **Lifetimes Overlap:** The lifetimes of the two allocations overlap. This is true if the maximum of their start times is less than or equal to the minimum of their end times:

        .. math::
           \max(T_{start\_current}, T_{start\_placed}) \le \min(T_{end\_current}, T_{end\_placed})

    b.  **Memory Regions Overlap:** The memory region the current allocation would occupy at ``current_offset`` (:math:`[\mbox{current_offset}, \mbox{current_offset} + S_{current})`) overlaps with the memory region of the placed allocation (:math:`[O_{placed}, O_{placed} + S_{placed})`). This is true if they are *not* entirely disjoint:

        .. math::
           \neg ( (\mbox{current_offset} + S_{current} \le O_{placed}) \lor (O_{placed} + S_{placed} \le \mbox{current_offset}) )

    If a conflict is found with any placed allocation, the current ``current_offset`` is not suitable. The heuristic then moves to check the next potential offset.

    **Offset Jumping Optimization:** A key optimization in this heuristic is that if a conflict is found with a placed allocation, the next ``current_offset`` to check is jumped forward. Instead of just incrementing ``current_offset`` by 1, the heuristic considers moving the ``current_offset`` past the end of the conflicting placed allocation (:math:`O_{placed} + S_{placed}`). The ``current_offset`` is updated to be the maximum of the default increment and the end of *any* conflicting placed allocation encountered at the current ``current_offset``. This helps to quickly skip over occupied memory blocks.

    .. math::
       \mbox{next_offset} = \max(\mbox{current_offset} + 1, \max_{conflicting\_placed} (O_{placed} + S_{placed}))

    (where the max is taken over all placed allocations that conflict at :math:`\mbox{current_offset}`)

    The heuristic continues checking offsets until it finds one where no conflicts exist with any placed allocation whose lifetime overlaps with the current allocation. This is the ``found_offset``.

4.  **Assign Offset and Add to Placed:** Once a ``found_offset`` is determined, it is assigned to the current allocation, and the allocation is added to the list of ``placed_allocations``.

5.  **Repeat:** Steps 2-4 are repeated for all allocations in the sorted order.

Tracing the Heuristic for the Example Data
-------------------------------------------

Let's trace the steps for our ``allocations_data``:

Sorted Order: Alloc 5 (12, [5, 9]), Alloc 0 (10, [1, 5]), Alloc 2 (8, [1, 3]), Alloc 4 (6, [3, 8]), Alloc 1 (5, [2, 6]), Alloc 3 (4, [4, 7]).

1.  **Place Alloc 5 (size 12, [5, 9]):** Offset = 0. Uses [0, 12) during [5, 9].
2.  **Place Alloc 0 (size 10, [1, 5]):** Conflicts with Alloc 5 at 0. First fit is at Offset 12. Uses [12, 22) during [1, 5].
3.  **Place Alloc 2 (size 8, [1, 3]):** Conflicts with Alloc 0 at 12. No conflict with Alloc 5 based on lifetime. First fit is at Offset 0. Uses [0, 8) during [1, 3].
4.  **Place Alloc 4 (size 6, [3, 8]):** Conflicts with Alloc 5 at 0, Alloc 0 at 12, Alloc 2 at 0. First fit is at Offset 22. Uses [22, 28) during [3, 8].
5.  **Place Alloc 1 (size 5, [2, 6]):** Conflicts with Alloc 5 at 0, Alloc 0 at 12, Alloc 2 at 0, Alloc 4 at 22. First fit is at Offset 28. Uses [28, 33) during [2, 6].
6.  **Place Alloc 3 (size 4, [4, 7]):** Conflicts with Alloc 5 at 0, Alloc 0 at 12, Alloc 4 at 22, Alloc 1 at 28. First fit is at Offset 33. Uses [33, 37) during [4, 7].

Heuristic Offsets (by original ID):

- Allocation 0: Offset = 12
- Allocation 1: Offset = 28
- Allocation 2: Offset = 0
- Allocation 3: Offset = 33
- Allocation 4: Offset = 22
- Allocation 5: Offset = 0

Calculating Peak Memory Usage
-----------------------------

Now, we calculate the peak memory usage with these offsets by checking the maximum memory used at each relevant time point (1, 2, 3, 4, 5, 6, 7, 8, 9):

- Time 1: Active: Alloc 0 ([12, 22)), Alloc 2 ([0, 8)). Max end: :math:`\max(22, 8) = 22`.
- Time 2: Active: Alloc 0 ([12, 22)), Alloc 1 ([28, 33)), Alloc 2 ([0, 8)). Max end: :math:`\max(22, 33, 8) = 33`.
- Time 3: Active: Alloc 0 ([12, 22)), Alloc 1 ([28, 33)), Alloc 2 ([0, 8)), Alloc 4 ([22, 28)). Max end: :math:`\max(22, 33, 8, 28) = 33`.
- Time 4: Active: Alloc 0 ([12, 22)), Alloc 1 ([28, 33)), Alloc 4 ([22, 28)), Alloc 3 ([33, 37)). Max end: :math:`\max(22, 33, 28, 37) = 37`.
- Time 5: Active: Alloc 0 ([12, 22)), Alloc 1 ([28, 33)), Alloc 4 ([22, 28)), Alloc 3 ([33, 37)), Alloc 5 ([0, 12)). Max end: :math:`\max(22, 33, 28, 37, 12) = 37`.
- Time 6: Active: Alloc 1 ([28, 33)), Alloc 4 ([22, 28)), Alloc 3 ([33, 37)), Alloc 5 ([0, 12)). Max end: :math:`\max(33, 28, 37, 12) = 37`.
- Time 7: Active: Alloc 4 ([22, 28)), Alloc 3 ([33, 37)), Alloc 5 ([0, 12)). Max end: :math:`\max(28, 37, 12) = 37`.
- Time 8: Active: Alloc 4 ([22, 28)), Alloc 5 ([0, 12)). Max end: :math:`\max(28, 12) = 28`.
- Time 9: Active: Alloc 5 ([0, 12)). Max end: :math:`12`.

The peak memory usage is the maximum of these values, which is **37**.

Visualization
-------------

We can visualize the allocation schedule below.

.. image:: ./memory_allocations.png

This entire test can be reproduced locally by running the following command.


.. code-block:: python

   pytest -s tests/kernel/wave/memory_test.py
