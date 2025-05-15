# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ....support.logging import get_logger

logger = get_logger("wave.memory_analysis.solver")


def determine_allocations_offsets(
    allocations_data: list[tuple[int, int, int]],
) -> list[int]:
    # Sort allocations by size in decreasing order
    allocations_data = [(i,) + x for i, x in enumerate(allocations_data)]
    sorted_allocations = sorted(allocations_data, key=lambda x: x[1], reverse=True)
    # Allocate memory for each allocation
    placed_allocations = []
    allocation_offsets = [None] * len(allocations_data)
    for i, size, start_time, end_time in sorted_allocations:
        # Find first available memory slot
        offset = -1
        # The starting offset for the current allocation.
        current_offset = 0
        # Determines whether the current slot is conflicting with any placed allocation.
        while offset == -1:

            next_offset = current_offset + 1
            memory_conflict = False

            for (
                placed_size,
                placed_start_time,
                placed_end_time,
                placed_offset,
                placed_size,
            ) in placed_allocations:

                # If the lifetimes overlap, then we need to make sure the current allocation
                # is not placed in the same memory slot.
                if max(start_time, placed_start_time) <= min(end_time, placed_end_time):
                    memory_overlap = not (
                        placed_offset + placed_size <= current_offset
                        or current_offset + size <= placed_offset
                    )
                    if memory_overlap:
                        memory_conflict = True
                        next_offset = max(next_offset, placed_offset + placed_size)

            if not memory_conflict:
                offset = current_offset
            else:
                current_offset = next_offset

        allocation_offsets[i] = offset
        placed_allocations.append((size, start_time, end_time, offset, size))

    # Determine how much total memory to allocate.
    allocation_size = max(offset + size for _, _, _, offset, size in placed_allocations)
    return allocation_offsets, allocation_size
