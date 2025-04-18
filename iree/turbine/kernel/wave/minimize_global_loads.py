# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexingContext, IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
from .utils.general_utils import (
    delinearize_index,
    ceildiv,
    is_shared_read,
    get_fastest_index,
)
from .utils.graph_utils import (
    DCE,
)
from .utils.symbol_utils import (
    subs_idxc,
)
from math import prod
import torch.fx as fx
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from ..lang.wave_types import IndexMapping


@dataclass
class SharedReadMetadata:
    index: dict[IndexSymbol, IndexSequence]
    mapping: IndexMapping
    memory_shape: tuple[int | IndexExpr]


def has_write_shared_user(node: Read) -> bool:
    return any(
        isinstance(user, Write)
        and subs_idxc(user.memory_type.address_space) == SHARED_ADDRESS_SPACE
        for user in node.users
    )


def is_valid_global_read(node: fx.Node) -> bool:
    custom = get_custom(node)
    return (
        isinstance(custom, Read)
        and subs_idxc(custom.memory_type.address_space) == GLOBAL_ADDRESS_SPACE
        and has_write_shared_user(custom)
    )


def is_transposed_read(custom: Read) -> bool:
    """
    Checks whether or not we are doing a transposed read.
    Returns true if the fastest dim in register is not
    the same as fastest dim in global memory.
    """
    assert isinstance(custom, Read) and "Expected input to be Read"
    global_fastest_dim = get_custom(custom.memory).type.symbolic_shape[-1]
    fastest_dim_idx = get_fastest_index(custom.index)
    register_fastest_dim = list(custom.index)[fastest_dim_idx]
    return register_fastest_dim != global_fastest_dim


def construct_min_global_access_pattern(
    index: dict[IndexSymbol, IndexSequence],
    thread_id: IndexExpr,
    load_elems_per_thread: int,
    shape: list[int],
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function constructs a new access pattern for a global read node.
    It retains workgroup and induction variable indexing but removes any thread
    level indexing which is inherited from the mma nodes during expansion.
    It takes a 1-D global offset and delinearizes it to a multi-dimensional offset
    and updates the access pattern accordingly.
    """
    thread_ids = [THREAD_0, THREAD_1, THREAD_2, GPR_NUM]
    new_index = {key: index[key].subs({t: 0 for t in thread_ids}) for key in index}
    nd_index = delinearize_index(thread_id, shape)
    for i, key in enumerate(index.keys()):
        new_index[key].start += nd_index[i]
        new_index[key].size = load_elems_per_thread if i == len(index.keys()) - 1 else 1
        new_index[key].stride = 1
    return new_index


def materialize_shape(
    constraint_tile_size: dict[IndexSymbol, int], symbolic_shape: list[IndexSymbol]
) -> list[int]:
    materialized_shape = []
    for dim in symbolic_shape:
        if dim in constraint_tile_size:
            materialized_shape.append(subs_idxc(constraint_tile_size[dim]))
        else:
            materialized_shape.append(subs_idxc(dim))
    return materialized_shape


def update_index_dims(src_node: fx.Node, target_expr: IndexExpr):
    """
    Given src node and a dst index expression, if src node has dims that also exist in target_expr,
    we copy the expression in target_expr into src node's index expression.
    """
    src_keys = set(src_node.index.keys())
    target_keys = set(target_expr.keys())
    intersection_keys = src_keys & target_keys
    for intersect_key in intersection_keys:
        src_node.index[intersect_key] = target_expr[intersect_key]


def identify_optimizable_loads(
    global_read_nodes: list[fx.Node],
    constraint_tile_size: dict[IndexSymbol, int],
    load_elems_per_thread: int,
    max_elements_per_load: int,
    allow_dynamic_transposed: bool = False,
    use_memory_type: bool = False,
) -> list[Read]:
    """
    Identify sub-optimal global loads that can be removed. A given memory has
    sub-optimal global loads if
        num_global_loads > (M * N) / (T * L)
    where the memory has shape [M, N], there are T threads and each thread can load L elements.
    """
    optimizable_loads: dict[fx.Node, tuple[int, list[Read], set["Custom"]]] = {}
    processed_memories = set()
    for read_node in global_read_nodes:
        custom = get_custom(read_node)
        if custom.memory in processed_memories:
            if custom.memory in optimizable_loads:
                optimizable_loads[custom.memory][1].append(custom)
            continue

        # TODO: We need to properly update index/elements_per_thread on dependent reads.
        if is_transposed_read(custom) and len(custom.mapping_dynamic_vals) > 0:
            if not allow_dynamic_transposed:
                continue

        processed_memories.add(custom.memory)
        symbolic_shape = custom.type.symbolic_shape
        if use_memory_type:
            symbolic_shape = custom.memory_type.symbolic_shape
        materialized_shape = materialize_shape(constraint_tile_size, symbolic_shape)
        # Ensure that the innermost dimension of the shape is a multiple of the elements being loaded.
        if materialized_shape[-1] % max_elements_per_load == 0:
            continue

        total_number_of_elements = prod(materialized_shape)
        expected_number_of_loads = ceildiv(
            total_number_of_elements, max_elements_per_load
        )
        actual_number_of_loads = len(
            [x for x in global_read_nodes if get_custom(x).memory == custom.memory]
        )
        if expected_number_of_loads >= actual_number_of_loads:
            continue

        expanded_dynamic_vals = None
        memory_load_elems_per_thread = load_elems_per_thread
        memory_max_elements_per_load = max_elements_per_load
        if len(custom.mapping_dynamic_vals) > 0 and not allow_dynamic_transposed:
            expanded_dynamic_vals = set(
                [
                    get_custom(user).mapping_dynamic_vals
                    for user in custom.memory.users.keys()
                ]
            )
            expanded_dynamic_vals = list(expanded_dynamic_vals)
            num_dynamic_vals = len(expanded_dynamic_vals)
            if num_dynamic_vals == expected_number_of_loads:
                pass
            elif expected_number_of_loads > 1 and num_dynamic_vals == 1:
                # If only one dyn val and many loads, broadcast dynamic vals across all the loads.
                # We would need actual copies instead of same reference, because each copy will have
                # it's own unique offset from the minimum_global_access_pattern.
                for i in range(1, expected_number_of_loads):
                    cur_expanded_dyn_vals = [
                        get_custom(dyn_val).copy(anchor=(dyn_val)).fx_node
                        for dyn_val in expanded_dynamic_vals[0]
                    ]
                    expanded_dynamic_vals.append(cur_expanded_dyn_vals)
            elif (
                num_dynamic_vals > 1
                and expected_number_of_loads == 1
                and load_elems_per_thread % num_dynamic_vals == 0
            ):
                # If expected one loads but many dyn val, break apart the load to as many dyn val.
                expected_number_of_loads = num_dynamic_vals
                memory_load_elems_per_thread //= num_dynamic_vals
                memory_max_elements_per_load //= num_dynamic_vals
            else:
                # Optimization do not handle other cases than above, so skip.
                continue

        optimizable_loads[custom.memory] = (
            expected_number_of_loads,
            [custom],
            expanded_dynamic_vals,
            memory_load_elems_per_thread,
            memory_max_elements_per_load,
        )
    return optimizable_loads


def add_optimized_nodes(
    optimizable_loads: dict[fx.Node, tuple[int, Read]],
    constraint_tile_size: dict[IndexSymbol, int],
    hardware_constraint: HardwareConstraint,
) -> dict[fx.Node, list[fx.Node]]:
    """
    Add optimized global read nodes and shared write nodes to the graph.
    """
    optimized_writes = defaultdict(list)
    for memory, (
        expected_number_of_loads,
        custom_loads,
        expanded_dynamic_vals,
        load_elems_per_thread,
        max_elements_per_load,
    ) in optimizable_loads.items():
        access_pattern: dict[IndexSymbol, IndexSequence] = custom_loads[0].index
        custom = custom_loads[0]
        if expanded_dynamic_vals:
            assert len(expanded_dynamic_vals) == expected_number_of_loads
        for i in range(expected_number_of_loads):
            with custom.graph.inserting_before(custom.fx_node):
                read = Read(memory, load_elems_per_thread, custom.mapping).add_to_graph(
                    custom.graph
                )
                global_offset = (
                    hardware_constraint.linearized_thread_id * load_elems_per_thread
                    + i * max_elements_per_load
                )
                materialized_shape = materialize_shape(
                    constraint_tile_size, custom.type.symbolic_shape
                )
                read.index = construct_min_global_access_pattern(
                    access_pattern,
                    global_offset,
                    load_elems_per_thread,
                    materialized_shape,
                )
                if custom.mapping_dynamic_vals:
                    # Update the dynamic vals' index expressions to match the min global access patterns.
                    for dyn_val in expanded_dynamic_vals[i]:
                        update_index_dims(dyn_val, read.index)
                    get_custom(read).update_arg(
                        "mapping_dynamic_vals", expanded_dynamic_vals[i]
                    )
                for custom_user in custom.users:
                    if (
                        isinstance(custom_user, Write)
                        and custom_user.type.address_space == SHARED_ADDRESS_SPACE
                    ):
                        write = Write(
                            read, custom_user.memory, load_elems_per_thread
                        ).add_to_graph(custom.graph)
                        write.index = read.index
                        optimized_writes[custom_user.memory].append(write)
                        write.vector_shapes = custom.vector_shapes
                        break
    return optimized_writes


def update_shared_memory_read(
    writes: list[fx.Node], shared_read_metadata: SharedReadMetadata, shared_read: Read
):
    """
    This function updates the shared memory reads as follows:
    - The index is reordered as per the original index.
    - The mapping is updated to the new mapping.
    - The shape of the alloc is updated to account for the fact that we are doing a gather.

    """
    assert writes[0] in shared_read_metadata, f"Write {writes[0]} not found in metadata"
    metadata = shared_read_metadata[writes[0]]
    original_index = deepcopy(shared_read.index)
    # Keep the shared read index, but re-order it like the original index.
    shared_read.index = {}
    for dim in metadata.index:
        shared_read.index[dim] = original_index[dim]
    # Apply the new mapping to the shared read.
    shared_read.update_arg("mapping", metadata.mapping)
    # If we are doing a gather from shared memory, we need to update the
    # shape of the alloc as well.
    custom_memory = get_custom(shared_read.memory)
    custom_memory_shape = custom_memory.type.symbolic_shape
    if custom_memory_shape != metadata.memory_shape:
        permutation = [custom_memory_shape.index(k) for k in metadata.memory_shape]
        custom_memory.update_arg("shape", metadata.memory_shape)
        new_distributed_shape = []
        for i, perm in enumerate(permutation):
            offset = 0
            if perm == len(permutation) - 1:
                offset = -custom_memory.padding
            elif i == len(permutation) - 1:
                offset = custom_memory.padding
            new_distributed_shape.append(custom_memory.distributed_shape[perm] + offset)
        custom_memory.update_arg("distributed_shape", tuple(new_distributed_shape))


def update_write_dependencies(
    optimized_writes: list[fx.Node],
    trace: CapturedTrace,
    shared_read_metadata: SharedReadMetadata = None,
):
    """
    Update all read shared nodes that have write dependencies on the unoptimized writes to
    the new optimized writes.
    """
    for memory, writes in optimized_writes.items():

        def is_replaceable_write(node: fx.Node) -> bool:
            custom = get_custom(node)
            return (
                isinstance(custom, Write)
                and custom.memory == memory
                and custom.type.address_space == SHARED_ADDRESS_SPACE
                and not custom.fx_node in writes
            )

        replaceable_writes = trace.walk(is_replaceable_write)
        for replaceable_write in replaceable_writes:
            for user in replaceable_write.users:
                idx = user.args.index([replaceable_write])
                custom_user = get_custom(user)
                custom_user.update_arg(idx, writes)
                if is_shared_read(custom_user) and shared_read_metadata:
                    update_shared_memory_read(writes, shared_read_metadata, custom_user)
                break

    DCE(trace)


def minimize_global_loads(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function attempts to minimize the number of global loads in a graph.
    If we have to load a tensor of shape [.., N] and we have T
    threads and each thread can load a maximum of L elements, then as long
    as N % L == 0, we can load the entire tensor with ceil(prod([.., N]) / (T * L)) global loads.
    This function applies this transformation as long as the condition above holds.

    """
    global_read_nodes = trace.walk(is_valid_global_read)
    if not global_read_nodes:
        return

    hardware_constraint = next(
        c for c in constraints if isinstance(c, HardwareConstraint)
    )
    constraint_tile_size = {
        c.dim: c.tile_size
        for c in constraints
        if isinstance(c, TilingConstraint) or isinstance(c, WorkgroupConstraint)
    }

    total_number_of_threads = hardware_constraint.threads_per_wave * prod(
        hardware_constraint.waves_per_block
    )
    element_type = get_custom(global_read_nodes[0]).type.dtype
    load_elems_per_thread = hardware_constraint.max_elems_per_load(element_type)
    max_elements_per_load = total_number_of_threads * load_elems_per_thread

    optimizable_loads = identify_optimizable_loads(
        global_read_nodes,
        constraint_tile_size,
        load_elems_per_thread,
        max_elements_per_load,
    )

    # Construct new global read nodes and write shared nodes.
    optimized_writes = add_optimized_nodes(
        optimizable_loads,
        constraint_tile_size,
        hardware_constraint,
    )

    # Update all write dependencies.
    update_write_dependencies(optimized_writes, trace)
