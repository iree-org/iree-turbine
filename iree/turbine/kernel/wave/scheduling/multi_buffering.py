from __future__ import annotations
from ...compiler.ir import (
    builtin_d,
    InsertionPoint,
    Location,
    Operation,
    transform_d,
    UnitAttr,
    Value,
)
from typing import Optional, Callable, Any, List, Tuple, Sequence
from ..._support.tracing import CapturedTrace
from ..._support.indexing import (
    IndexExpr,
    IndexingContext,
    IndexSymbol,
    IndexSequence,
    xor,
)
from ...lang.global_symbols import *
from ...ops.wave_ops import (
    get_custom,
    Output,
    Read,
    Write,
    MMA,
    CustomOp,
    Reduction,
    GetResult,
    ExtractSlice,
    IterArg,
    Reshape,
)
from ...lang.wave_types import IndexMapping
from ..constraints import (
    Constraint,
    WorkgroupConstraint,
    HardwareConstraint,
    TilingConstraint,
    MMAType,
    MMAOperand,
)
from ..assumptions import Assumption
from ..utils import print_trace
import torch.fx as fx
import iree.turbine.kernel.lang as tkl
from pathlib import Path


import tempfile
from ....support.conversions import TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM
from iree.compiler.dialects.transform import (
    interpreter as transform_interpreter,
    any_op_t,
)
from iree.compiler.dialects import (
    _structured_transform_ops_gen as structured_transform_ops,
)

K = tkl.sym.K
M = tkl.sym.M
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_K = tkl.sym.BLOCK_K
ARGK = tkl.IndexSymbol("$ARGK", integer=True, nonnegative=True)


def partition_by_memory(reads: list[CustomOp]) -> dict[CustomOp, list[CustomOp]]:
    """
    Partitions reads by their source memory location.
    Returns a dict mapping memory nodes to lists of read operations from that memory.
    """
    memory_to_reads: dict[CustomOp, list[CustomOp]] = {}

    for read_node in reads:
        memory_node = get_custom(read_node.memory)

        if memory_node not in memory_to_reads:
            memory_to_reads[memory_node] = []

        memory_to_reads[memory_node].append(read_node)

    return memory_to_reads


def multi_buffer(trace: CapturedTrace):
    """ """

    # Find all reductions
    reductions = trace.walk(lambda node: isinstance(get_custom(node), Reduction))

    # Get reduction dimension from first reduction
    if not reductions:
        return
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
    memory_to_reads = partition_by_memory(reads)
    memory_to_writes = partition_by_memory(writes)

    for memory_location in set(memory_to_reads.keys()) | set(memory_to_writes.keys()):
        read_nodes = memory_to_reads.get(memory_location, [])
        write_nodes = memory_to_writes.get(memory_location, [])

        implement_double_buffering(
            trace, memory_location, read_nodes, write_nodes, reduction_axis
        )


def implement_double_buffering(
    trace: CapturedTrace,
    original_buffer: CustomOp,
    read_nodes: list[Read],
    write_nodes: list[Write],
    axis: IndexSymbol,
):
    """
    Implements double buffering for a shared memory buffer.
    """
    # For now only double buffering, so we are doubling the
    # size of the original buffer
    assert len(original_buffer.shape) == 2

    # double the memory in the non-reduction dimension
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

    # For each read/write operation, modify its index to include buffer offset
    stage_mapping: dict[int, list[CustomOp]] = {}
    for custom_op in read_nodes + write_nodes:
        cycle = custom_op.fx_node.scheduling_parameters["cycle"]
        # TODO TODO TODO: Should this be cycle or stage?
        #                 With cycle i generate the code I want

        # Group nodes by their cycle
        if cycle not in stage_mapping:
            stage_mapping[cycle] = []
        stage_mapping[cycle].append(custom_op)

    for stage in stage_mapping.keys():
        print(f"modifying for {stage}")
        offset = 0
        for op in stage_mapping[stage]:
            buffer_offset = (ARGK % 2) * block_size
            #  TODO TODO TODO: This is still hardcoded
            if stage < 2:
                offset = buffer_offset
            elif stage >= 2:
                offset = xor(buffer_offset, block_size)
            else:
                raise CodegenError(
                    f"Stage > 4 not supported in multibuffering currently"
                )

            op.index[original_dim].start = op.index[original_dim].start + offset
