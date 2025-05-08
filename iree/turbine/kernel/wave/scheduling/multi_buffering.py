# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from ..._support.tracing import CapturedTrace
from ..._support.indexing import (
    IndexSymbol,
    xor,
)
from ...compiler.base import CodegenError
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...lang.wave_types import IndexMapping
from ...ops.wave_ops import (
    get_custom,
    Read,
    Write,
    CustomOp,
    Iterate,
)
from ..utils.mapping_utils import get_dict_with_updated_key
import iree.turbine.kernel.lang as tkl


def multi_buffer(trace: CapturedTrace):
    """Perform multi buffering for all supported shared memory locations"""

    # Find all reductions
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Iterate))

    # Get reduction dimension from first reduction
    if not reductions or len(reductions) != 1:
        raise CodegenError(
            f"Unexpected number of reductions found in graph: {len(reductions)} vs 1"
        )

    reduction_axis = get_custom(reductions[0]).axis

    # Find reads and writes operating on shared memory
    reads = []
    writes = []
    for node in trace.get_subgraph(get_custom(reductions[0]).subgraph_name).nodes:
        custom = get_custom(node)
        if (
            isinstance(custom, Read | Write)
            and custom.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            if isinstance(custom, Read):
                reads.append(custom)
            elif isinstance(custom, Write):
                writes.append(custom)

    # Partition reads and writes by memory location
    memory_to_reads = _partition_by_memory(reads)
    memory_to_writes = _partition_by_memory(writes)

    # Perform multi buffering for all collected memory locations
    for memory_location in set(memory_to_reads.keys()) | set(memory_to_writes.keys()):
        read_nodes = memory_to_reads.get(memory_location, [])
        write_nodes = memory_to_writes.get(memory_location, [])

        _multi_buffer_memory_location(
            trace, memory_location, read_nodes, write_nodes, reduction_axis, 2
        )


def _multi_buffer_memory_location(
    trace: CapturedTrace,
    original_buffer: CustomOp,
    read_nodes: list[Read],
    write_nodes: list[Write],
    reduction_axis: IndexSymbol,
    buffer_count: int,
):
    """
    Implements multi buffering for all reads and write of a shared memory buffer.
    """
    # For now we only support double buffering
    if buffer_count != 2:
        raise CodegenError(
            "Current multi buffering implementation supports only buffer_count=2"
        )

    # Add the buffer offset to the index of each read/write operation
    stage_mapping: dict[int, list[CustomOp]] = {}
    for custom_op in read_nodes + write_nodes:
        cycle = custom_op.fx_node.scheduling_parameters["cycle"]

        # Group nodes by their cycle
        if cycle not in stage_mapping:
            stage_mapping[cycle] = []
        stage_mapping[cycle].append(custom_op)

    reduction_dim_indices = [
        i for i, dim in enumerate(original_buffer.shape) if dim == reduction_axis
    ]
    induction_var = tkl.IndexSymbol(
        f"$ARG{reduction_axis.name}", integer=True, nonnegative=True
    )
    buffer_selector = induction_var % buffer_count  # 0 to buffer_count-1
    for stage in stage_mapping.keys():
        offset = 0
        for op in stage_mapping[stage]:
            # Determine buffer offset based on stage
            use_alternate_buffer = stage >= 2 and stage <= 4

            # Update each non-reduction dimension with appropriate offset
            for i, dim in enumerate(original_buffer.shape):
                if i not in reduction_dim_indices and dim in op.index:
                    block_size = original_buffer.distributed_shape[i]

                    # Calculate offset based on buffer selection
                    if use_alternate_buffer:
                        # XOR with block_size for ping-pong effect
                        offset = xor(buffer_selector * block_size, block_size)
                    else:
                        offset = buffer_selector * block_size
                    op.index[dim].start = op.index[dim].start + offset

                    # Update the mapping for the operation as the keys for the
                    # mapping have to match the shape of memory location the
                    # operation reads from / writes to, which we change below.
                    if isinstance(op.mapping, IndexMapping):
                        input_mapping = op.mapping.input_mapping
                        output_mapping = op.mapping.output_mapping
                        if dim in input_mapping:
                            op.mapping.input_mapping = get_dict_with_updated_key(
                                input_mapping, dim, dim * buffer_count
                            )
                            input_mapping = op.mapping.input_mapping
                        if dim in output_mapping:
                            op.mapping.output_mapping = get_dict_with_updated_key(
                                output_mapping, dim, dim * buffer_count
                            )
                            output_mapping = op.mapping.output_mapping

    # Create new shape with increased non-reduction dimensions
    new_shape = []
    new_distributed_shape = []

    for i, dim in enumerate(original_buffer.shape):
        if i in reduction_dim_indices:
            # Keep reduction dimensions as is
            new_shape.append(dim)
            new_distributed_shape.append(original_buffer.distributed_shape[i])
        else:
            # Increase non-reduction dimensions
            new_shape.append(dim * buffer_count)
            new_distributed_shape.append(
                original_buffer.distributed_shape[i] * buffer_count
            )

    original_buffer.update_arg(0, tuple(new_shape))
    original_buffer.update_arg(1, tuple(new_distributed_shape))


def _partition_by_memory(nodes: list[CustomOp]) -> dict[CustomOp, list[CustomOp]]:
    """
    Partitions reads or writes by their source memory location.
    Returns a dict mapping memory nodes to lists of read or write operations from that memory.
    """
    memory_mapping: dict[CustomOp, list[CustomOp]] = {}

    for node in nodes:
        memory_node = get_custom(node.memory)

        if memory_node not in memory_mapping:
            memory_mapping[memory_node] = []

        memory_mapping[memory_node].append(node)

    return memory_mapping
