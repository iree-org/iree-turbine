from __future__ import annotations
from ..._support.tracing import CapturedTrace
from ..._support.indexing import (
    IndexSymbol,
    xor,
)
from ...compiler.base import CodegenError
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
from ...ops.wave_ops import (
    get_custom,
    Read,
    Write,
    CustomOp,
    Reduction,
)
import iree.turbine.kernel.lang as tkl


def multi_buffer(trace: CapturedTrace):
    """Perform multi buffering for all supported shared memory locations"""

    # Find all reductions
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Reduction))

    # Get reduction dimension from first reduction
    if not reductions or len(reductions) != 1:
        raise CodegenError(
            f"Unexpected number of reductions found in graph: {len(reductions)} vs 1"
        )

    reduction_axis = get_custom(reductions[0]).axis

    # Find reads that index using the reduction dim and are from shared memory
    reads = []
    writes = []
    for node in trace.get_subgraph(get_custom(reductions[0]).subgraph_name).nodes:
        custom = get_custom(node)
        if not hasattr(custom, "memory_type"):
            continue
        if (
            reduction_axis in custom.indexing_dims
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
    axis: IndexSymbol,
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

    # double the memory in the non-reduction dimension
    if len(original_buffer.shape) != 2:
        raise CodegenError(
            "Current multi buffering implementation supports only reads/writes with size 2"
        )
    reduction_dim_index = original_buffer.shape.index(axis)
    original_dim = original_buffer.shape[1 - reduction_dim_index]

    block_size = original_buffer.distributed_shape[1 - reduction_dim_index]
    new_shape = tuple(
        dim * 2 if i != reduction_dim_index else dim
        for i, dim in enumerate(original_buffer.shape)
    )
    new_distributed_shape = tuple(
        dim * 2 if i != reduction_dim_index else dim
        for i, dim in enumerate(original_buffer.distributed_shape)
    )
    original_buffer.update_arg(0, new_shape)
    original_buffer.update_arg(1, new_distributed_shape)

    # Add the buffer offset to the index of each read/write operation
    stage_mapping: dict[int, list[CustomOp]] = {}
    for custom_op in read_nodes + write_nodes:
        cycle = custom_op.fx_node.scheduling_parameters["cycle"]

        # Group nodes by their cycle
        if cycle not in stage_mapping:
            stage_mapping[cycle] = []
        stage_mapping[cycle].append(custom_op)

    induction_var = tkl.IndexSymbol(f"$ARG{axis.name}", integer=True, nonnegative=True)
    for stage in stage_mapping.keys():
        offset = 0
        for op in stage_mapping[stage]:
            buffer_offset = (induction_var % 2) * block_size
            if stage < 2:
                offset = buffer_offset
            elif stage >= 2 and stage <= 4:
                offset = xor(buffer_offset, block_size)
            else:
                raise CodegenError(
                    "The current multibuffering implementation does not support read/write with Stage > 4."
                )

            op.index[original_dim].start = op.index[original_dim].start + offset


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
