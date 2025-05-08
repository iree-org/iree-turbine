# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import unittest
from iree.turbine.kernel.wave.memory_analysis.solver import (
    determine_allocations_offsets,
)
from iree.turbine.kernel.wave.memory_analysis.visualize import (
    visualize_memory_allocations,
)


class MemoryTest(unittest.TestCase):
    def test_memory_allocation(self):
        # Note: Times are discrete steps. An allocation is active from start_time to end_time inclusive.
        visualize = False
        allocations_data = [
            (10, 1, 5),  # Allocation 0: size 10, active from time 1 to 5
            (5, 2, 6),  # Allocation 1: size 5, active from time 2 to 6
            (8, 1, 3),  # Allocation 2: size 8, active from time 1 to 3
            (4, 4, 7),  # Allocation 3: size 4, active from time 4 to 7
            (6, 3, 8),  # Allocation 4: size 6, active from time 3 to 8
            (12, 5, 9),  # Allocation 5: size 12, active from time 5 to 9
        ]

        (
            heuristic_offsets,
            heuristic_size,
        ) = determine_allocations_offsets(allocations_data)
        if visualize:
            visualize_memory_allocations(
                allocations_data,
                heuristic_offsets,
                heuristic_size,
                "memory_allocations.png",
            )

        assert heuristic_size == 37
        assert heuristic_offsets == [12, 28, 0, 33, 22, 0]
