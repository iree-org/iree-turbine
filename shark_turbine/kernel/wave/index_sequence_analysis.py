# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ops.wave_ops import Write, ExtractSlice, get_custom
from .constraints import Constraint, HardwareConstraint
from .._support.tracing import CapturedTrace, IndexingContext
from .._support.indexing import IndexSymbol, IndexSequence
from ..lang.global_symbols import *
from .utils import (
    simplify_index,
    get_mma_dimensional_mapping,
    get_hardware_vector_size,
    subs_idxc,
)
import torch.fx as fx
import numpy as np


def get_vector_shape(
    trace: CapturedTrace,
    hardware_constraint: HardwareConstraint,
    symbolic_shape: list[IndexSymbol],
) -> list[int]:
    mma_indices, _ = get_mma_dimensional_mapping(trace)
    return [
        get_hardware_vector_size(dim, hardware_constraint, mma_indices)
        for dim in symbolic_shape
    ]


def partition_strided_operators(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of operators in the graph
    that are writes on 2d tensors. If the operator has an access pattern where
    the strides are greater than one on a single dimension, this function splits the
    operands into individual elements and constructs a write for
    each individual element.
    """

    def has_strided_access(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if isinstance(custom, Write) and len(custom.register_type.symbolic_shape) == 2:
            strides = [
                simplify_index(custom.register_index[dim]).stride
                for dim in custom.register_index
            ]
            elements_per_thread = [
                simplify_index(custom.register_index[dim]).size
                for dim in custom.register_index
            ]
            strides = [x for x, y in zip(strides, elements_per_thread) if y > 1]
            num_strided_accesses = sum(1 for stride in strides if stride > 1)
            if num_strided_accesses > 1:
                raise NotImplementedError(
                    "Support for strided accesses on more than one dimension not implemented yet!"
                )
            return num_strided_accesses == 1
        return False

    strided_operators = trace.walk(has_strided_access)
    hw_constraint = [c for c in constraints if isinstance(c, HardwareConstraint)][0]
    for operator in strided_operators:
        custom = get_custom(operator)
        simplified_index = {
            dim: simplify_index(custom.register_index[dim]) for dim in custom.index
        }

        max_stride = int(max(simplified_index[dim].stride for dim in simplified_index))
        shape = get_vector_shape(
            trace, hw_constraint, custom.register_type.symbolic_shape
        )
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        with custom.graph.inserting_before(operator):
            for i in range(elements_per_thread):
                extract = ExtractSlice(custom.register_, [i], [1], [1]).add_to_graph(
                    custom.graph
                )
                offset = np.unravel_index(i * max_stride, shape)
                write = Write(
                    extract,
                    custom.memory,
                    mapping=custom.mapping,
                    elements_per_thread=1,
                ).add_to_graph(custom.graph)
                write.index = {
                    dim: IndexSequence(simplified_index[dim].start + offset[j], 1, 1)
                    for j, dim in enumerate(custom.register_type.symbolic_shape)
                }
        custom.graph.erase_node(operator)
