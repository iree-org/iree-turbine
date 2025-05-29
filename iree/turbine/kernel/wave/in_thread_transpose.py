# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
from typing import Sequence
from ..wave.constraints import (
    Constraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..lang.wave_types import IndexMapping
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
from math import prod
import torch.fx as fx
from collections import defaultdict
from .utils.symbol_utils import safe_subs
from .utils.general_utils import (
    ceildiv,
    delinearize_index,
    get_hardware_constraint,
    remove_thread_indexing,
)
from .minimize_global_loads import (
    is_transposed_read,
    materialize_shape,
    update_write_dependencies,
)
from .global_to_shared_gathers import update_read_mapping_dynamic_values
from ..ops.wave_ops import Extract, Read, Write, Reshape
import logging

logger = logging.getLogger(__name__)


def is_transpose_read(node: fx.Node) -> bool:
    read = get_custom(node)
    if not isinstance(read, Read):
        return False

    src_shape = read.type.symbolic_shape
    if len(src_shape) <= 1:
        return False

    return is_transposed_read(read)


def combine_index(
    index1: dict[IndexSymbol, IndexSequence],
    index2: dict[IndexSymbol, IndexSequence],
    fastest_dim: IndexSymbol,
    fastest_dim_vec_size: int,
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes two index sequences and combines them.
    """
    assert len(index1) == len(index2)
    return {
        key: IndexSequence(
            index1[key].start + index2[key].start,
            fastest_dim_vec_size if key == fastest_dim else 1,
            1,
        )
        for key in index2
    }


def transpose_last2(shape: Sequence[IndexSymbol]) -> list[IndexSymbol]:
    return list(shape[:-2]) + [shape[-1], shape[-2]]


def get_tiled_shape(
    shape: Sequence[IndexSymbol], vec_size: int, num_vecs: int
) -> list[IndexSymbol]:
    """
    Given the source shape and expected tile size `num_vecs x vector<vec_size x DT>`
    return shape denoting number of tiles.
    """
    new_shape = list(shape)
    new_shape[-1] //= vec_size
    new_shape[-2] //= num_vecs
    return new_shape


def get_tiled_index(
    linear_id: IndexExpr,
    tiled_shape: Sequence[IndexSymbol],
    vec_size: int,
    num_vecs: int,
    vec_index: int,
    transpose: bool,
) -> list[IndexSymbol]:
    """
    Given tiled_shape and vec_size/num_vecs construct the index for each vector
    operation.
    """
    if transpose:
        index = delinearize_index(linear_id, transpose_last2(tiled_shape))
        index = transpose_last2(index)
    else:
        index = delinearize_index(linear_id, tiled_shape)

    index[-1] *= vec_size
    index[-2] = index[-2] * num_vecs + vec_index
    return index


def in_thread_transpose(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This pass is detecting when read-write pair is doing transpose and tries to
    combine it with minimize_global_loads optimization.

    i.e. if minimize_global_loads will result in 4 x vector<8xf16> stores to shared
    memory, we can do 8 x vector<4xf16> loads from global memory and do transpose
    in each thread registers, using sequence of vector.extract_strided_slice/insert_strided_slice.
    """
    logger.info("in_thread_transpose")
    candidates = trace.walk(is_transpose_read)
    for candidate in candidates:
        logger.info(candidate, candidate.index)

    mem_to_read_write = defaultdict(list)
    for read in candidates:
        read = get_custom(read)
        for write in read.users:
            if not isinstance(write, Write):
                continue

            mem_to_read_write[(read.memory, write.memory)].append((read, write))

    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, (TilingConstraint, WorkgroupConstraint))
    }

    hardware_constraint = get_hardware_constraint(constraints)
    total_number_of_threads = hardware_constraint.threads_per_wave * prod(
        hardware_constraint.waves_per_block
    )

    for reads_writes in mem_to_read_write.values():
        read, write = reads_writes[0]
        logger.info(f"processing read={read}, write={write}")

        if read.mapping_dynamic_vals:
            # TODO: While we do support dynamic val mappings functionally,
            # changing it access pattern causes perf degradation on paged decode
            # attention.
            # Need to modify minimize_global_loads to use access-pattern of
            # dynamic val mappings.
            continue

        element_type = read.type.dtype

        load_elems_per_thread = hardware_constraint.max_elems_per_load(element_type)
        max_elements_per_load = total_number_of_threads * load_elems_per_thread
        logger.info(
            f"load_elems_per_thread={load_elems_per_thread}, "
            f"max_elements_per_load={max_elements_per_load}"
        )

        dst_symbolic_shape = read.type.symbolic_shape

        materialized_shape = materialize_shape(constraint_tile_size, dst_symbolic_shape)
        if any(s > 1 for s in materialized_shape[:-2]) or any(
            s <= 1 for s in materialized_shape[-2:]
        ):
            logger.info(
                f"only last 2 dims transpose is supported, got {materialized_shape}"
            )
            continue

        src_symbolic_shape = transpose_last2(dst_symbolic_shape)

        total_number_of_elements = prod(materialized_shape)
        logger.info(
            f"src_shape={src_symbolic_shape}, "
            f"dst_shape={dst_symbolic_shape}, "
            f"materialized_shape={materialized_shape}, "
            f"total_elements={total_number_of_elements}"
        )

        expected_number_of_loads = ceildiv(
            total_number_of_elements, max_elements_per_load
        )
        actual_number_of_loads = len(reads_writes)
        logger.info(
            f"expected_number_of_loads={expected_number_of_loads}, actual_number_of_loads={actual_number_of_loads}"
        )
        if expected_number_of_loads <= 1:
            continue

        if materialized_shape[-2] % expected_number_of_loads != 0:
            continue

        linear_id = hardware_constraint.linearized_thread_id
        global_index = remove_thread_indexing(read.index)
        logger.info(
            f"global_index={global_index}, load_elems_per_thread={load_elems_per_thread}"
        )

        new_reads = []

        load_shape = get_tiled_shape(
            materialize_shape(constraint_tile_size, src_symbolic_shape),
            expected_number_of_loads,
            load_elems_per_thread,
        )
        logger.info(f"load_shape={load_shape}")

        # Construct new reads.
        for i in range(load_elems_per_thread):
            read_index = get_tiled_index(
                linear_id,
                load_shape,
                expected_number_of_loads,
                load_elems_per_thread,
                i,
                transpose=False,
            )
            read_index = {
                k: IndexSequence(v, 1, 1)
                for k, v in zip(src_symbolic_shape, read_index)
            }
            read_index = combine_index(
                global_index,
                read_index,
                src_symbolic_shape[-1],
                expected_number_of_loads,
            )
            logger.info(f"read_index={read_index}")
            mapping = read.mapping
            if mapping is not None:
                # As we are transposing read, update mapping to still be output
                # identity.
                out_mapping = {
                    k: IndexMapping.iterator(i)
                    for i, k in enumerate(src_symbolic_shape)
                }
                subs = {v: out_mapping[k] for k, v in mapping.output_mapping.items()}
                input_mapping = {
                    k: safe_subs(v, subs, simultaneous=True)
                    for k, v in mapping.input_mapping.items()
                }
                mapping = IndexMapping(
                    num_iterators=len(out_mapping),
                    inputs=input_mapping,
                    outputs=out_mapping,
                    dynamic_val_mappings=mapping.dynamic_val_mappings,
                )

            with read.graph.inserting_before(read.fx_node):
                new_read = Read(
                    read.memory,
                    expected_number_of_loads,
                    mapping=mapping,
                    mapping_dynamic_vals=read.mapping_dynamic_vals,
                ).add_to_graph(read.graph)
                new_read.index = read_index
                new_read_custom = get_custom(new_read)
                new_read_custom.infer_type()
                update_read_mapping_dynamic_values(new_read_custom)
                new_reads.append(new_read)

        new_writes = defaultdict(list)
        repacked = []

        store_shape = get_tiled_shape(
            materialized_shape, load_elems_per_thread, expected_number_of_loads
        )
        logger.info(f"store_shape={store_shape}")

        # Construct transpose.
        for i in range(expected_number_of_loads):
            with write.graph.inserting_before(write.fx_node):
                values = []
                for j in range(load_elems_per_thread):
                    value = Extract(new_reads[j], [i]).add_to_graph(write.graph)
                    values.append(value)

                value = Reshape(values, load_elems_per_thread).add_to_graph(write.graph)
                repacked.append(value)

        # Construct new writes.
        for i in range(expected_number_of_loads):
            store_index = get_tiled_index(
                linear_id,
                store_shape,
                load_elems_per_thread,
                expected_number_of_loads,
                i,
                transpose=True,
            )
            store_index = {
                k: IndexSequence(v, 1, 1)
                for k, v in zip(dst_symbolic_shape, store_index)
            }
            store_index = combine_index(
                global_index, store_index, dst_symbolic_shape[-1], load_elems_per_thread
            )
            logger.info(f"store_index={store_index}")
            with write.graph.inserting_before(write.fx_node):
                value = repacked[i]
                value.index = store_index

                new_write = Write(
                    value,
                    write.memory,
                    load_elems_per_thread,
                    mapping=write.mapping,
                    mapping_dynamic_vals=write.mapping_dynamic_vals,
                ).add_to_graph(write.graph)
                new_write.index = store_index
                new_writes[write.memory].append(new_write)

        update_write_dependencies(new_writes, trace)
