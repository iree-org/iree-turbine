# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass


@dataclass
class KernelLaunchInfo:
    grid: tuple[int] = None
    blocks: tuple[int] = None
    shared_memory_bytes: int = 0
    func_name: str = ""
    grid_str: str = ""
