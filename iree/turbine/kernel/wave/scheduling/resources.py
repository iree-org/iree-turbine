# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...lang.global_symbols import *
from ..utils import subs_idxc
from ...ops.wave_ops import (
    Read,
    Write,
    MMA,
    IterArg,
    Output,
    get_custom,
    CustomOp,
    CastOp,
)
import torch.fx as fx
from enum import Enum
import numpy as np


# This table contains the number of functional units available for each operation.
def get_available_resources() -> list[int]:
    resources = [GLOBAL_MEMORY_UNITS, SHARED_MEMORY_UNITS, MMA_UNITS]
    return np.array([int(subs_idxc(x)) for x in resources])


class Operation(Enum):
    READ_SHARED = "read_shared"
    WRITE_SHARED = "write_shared"
    READ_GLOBAL = "read_global"
    WRITE_GLOBAL = "write_global"
    MMA = "mma"
    ALU = "alu"
    VALU = "valu"
    SALU = "salu"
    NOOP = "noop"


# This table contains the cycles required to execute each operation.
delay_table = {
    Operation.READ_SHARED: READ_SHARED_DELAY,
    Operation.WRITE_SHARED: WRITE_SHARED_DELAY,
    Operation.READ_GLOBAL: READ_GLOBAL_DELAY,
    Operation.WRITE_GLOBAL: WRITE_GLOBAL_DELAY,
    Operation.MMA: MMA_DELAY,
    Operation.NOOP: 0,
}

# This table contains the resource usage for each operation.
# Operations can use more than one resource for more than one cycle.
resource_reservation_table = {
    Operation.READ_SHARED: np.array([[0, 1, 0]]),
    Operation.WRITE_SHARED: np.array([[0, 1, 0]]),
    Operation.READ_GLOBAL: np.array([[1, 0, 0]]),
    Operation.WRITE_GLOBAL: np.array([[1, 0, 0]]),
    Operation.MMA: np.array([[0, 0, 1]]),
    Operation.NOOP: np.array([[0, 0, 0]]),
}


def get_custom_operation_type(custom: CustomOp) -> Operation:
    if isinstance(custom, Read):
        return (
            Operation.READ_GLOBAL
            if custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
            else Operation.READ_SHARED
        )
    elif isinstance(custom, Write):
        return (
            Operation.WRITE_GLOBAL
            if custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
            else Operation.WRITE_SHARED
        )
    elif isinstance(custom, MMA):
        return Operation.MMA
    elif isinstance(custom, IterArg):
        return Operation.NOOP
    elif isinstance(custom, Output):
        return Operation.NOOP
    else:
        return None


def annotate_resource_usage(
    graph: fx.Graph,
) -> tuple[set[fx.Node], list[fx.Node], fx.Node]:
    ignore_nodes = set()
    iter_args = []
    output = None
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, Read):
            custom.rrt = (
                resource_reservation_table[Operation.READ_GLOBAL]
                if custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
                else resource_reservation_table[Operation.READ_SHARED]
            )
        elif isinstance(custom, Write):
            custom.rrt = (
                resource_reservation_table[Operation.WRITE_GLOBAL]
                if custom.memory_type.address_space == GLOBAL_ADDRESS_SPACE
                else resource_reservation_table[Operation.WRITE_SHARED]
            )
        elif isinstance(custom, MMA):
            custom.rrt = resource_reservation_table[Operation.MMA]
        elif isinstance(custom, (IterArg, CastOp)):
            iter_args.append(node)
            custom.rrt = resource_reservation_table[Operation.NOOP]
        elif isinstance(custom, Output):
            output = node
        else:
            ignore_nodes.add(node)
    return ignore_nodes, iter_args, output


def get_scheduling_mask(operation: Operation) -> int:
    """
    Returns the scheduling mask for the given operation.
    """
    match operation:
        case Operation.READ_GLOBAL:
            return int("0x20", 0)
        case Operation.WRITE_GLOBAL:
            return int("0x40", 0)
        case Operation.READ_SHARED:
            return int("0x100", 0)
        case Operation.WRITE_SHARED:
            return int("0x200", 0)
        case Operation.MMA:
            return int("0x8", 0)
        case Operation.ALU:
            return int("0x1", 0)
        case Operation.VALU:
            return int("0x2", 0)
        case Operation.SALU:
            return int("0x4", 0)
    return None
