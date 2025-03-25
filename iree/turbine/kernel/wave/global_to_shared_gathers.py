# Copyright 2025 The IREE Authors
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
from .._support.indexing import IndexSequence, IndexSymbol, IndexExpr
from ..lang.wave_types import IndexMapping
from ..ops.wave_ops import Read, Write, get_custom
from ..lang.global_symbols import *
from math import prod
import torch.fx as fx
from collections import defaultdict
from copy import deepcopy
from .utils.symbol_utils import subs_idxc
from .utils.general_utils import is_gather
from .minimize_global_loads import (
    has_write_shared_user,
    construct_min_global_access_pattern,
    materialize_shape,
    identify_optimizable_loads,
    update_write_dependencies,
    SharedReadMetadata,
)

"""
We are given N global gathers that are promoted to shared memory. This function
allows us to convert these to:
 - M contiguous loads from global memory, M writes to shared memory, and N gathers from shared memory.

Promotion allows us to shift the burden of gathers up the memory hierarchy, which can be beneficial for performance.

Given a gather with an access pattern and mapping of the form:
*** Global gather ***
index = {H_KV: h : 1 : 1, D_KV: d : 1 : 1, N_KV: n : 4 : 1}
mapping = input: {H_KV: i // 4, D_KV: j, N_KV: d0},
          output: {H_KV: i, D_KV: j, N_KV: k}
          dynamic_val: {N_KV: j}

where d0 is a dynamic value from another read of possibly more
than one element.

We transform this to (assuming D_KV is the contiguous dimension):
*** Global load ***
Linear access pattern along read dimension, permuted mapping.
index = {H_KV: h : 1 : 1, N_KV: n_new : 1 : 1, D_KV: d_new : 8 : 1}
mapping = input: {N_KV: d1, H_KV: j // 4, D_KV: k},
          output: {N_KV: i, H_KV: j, D_KV: k}
          dynamic_val: {N_KV: i}

The value of d1 is determined by reading from the n_new offset.

*** Shared store ***
Linear access pattern, identity mapping.
index = {H_KV: h_new : 1 : 1, D_KV: d_new : 8 : 1, N_KV: n_new : 1 : 1}
mapping = input: {H_KV: i, N_KV: j, D_KV: k},
          output: {H_KV: i, N_KV: j, D_KV: k}

*** Shared gather ***
MMA access pattern, identity mapping.
index = {H_KV: h : 1 : 1, D_KV: d : 1 : 1, N_KV: n : 4 : 1}
mapping = input: {H_KV: i , D_KV: j, N_KV: k},
          output: {H_KV: i, D_KV: j, N_KV: k}

TODO: Support load -> scatter -> load pattern
We can also modify this function to load contiguously from global memory, scatter to shared memory
and the load contiguously from global memory.

"""


def is_valid_global_gather(node: fx.Node) -> bool:
    custom = get_custom(node)
    return (
        isinstance(custom, Read)
        and subs_idxc(custom.memory_type.address_space) == GLOBAL_ADDRESS_SPACE
        and has_write_shared_user(custom)
        and is_gather(custom)
    )


def make_contiguous_index(read: Read) -> dict[IndexSymbol, IndexSequence]:
    """
    This function modifies the index so that we are loading the most
    elements along the contiguous dimension. This results in an invalid
    MMA access pattern but that is okay since we are modifying the access
    pattern from an MMA to linear access pattern.
    """
    contiguous_index = {}
    max_load_size = max(x.size for x in read.index.values())
    for i, dim in enumerate(read.memory_type.symbolic_shape):
        contiguous_index[dim] = read.index[dim]
        if i == len(read.memory_type.symbolic_shape) - 1:
            contiguous_index[dim].size = max_load_size
        else:
            contiguous_index[dim].size = 1
    return contiguous_index


def construct_global_read_mapping(read: Read) -> IndexMapping:
    """
    This function constructs a new mapping for the read node.
    """
    # Create new identity output mapping.
    mapping = read.mapping
    new_mapping = deepcopy(mapping)
    new_mapping.output_mapping = {}
    assert mapping.is_output_identity(), "Output mapping must be an identity mapping."
    for dim, iter in zip(read.memory_type.symbolic_shape, mapping.iters):
        new_mapping.output_mapping[dim] = iter

    # Find the iter permutation.
    old_to_new_iter = {}
    for dim in mapping.output_mapping:
        old_to_new_iter[mapping.output_mapping[dim]] = new_mapping.output_mapping[dim]

    # Create new input mapping with only dynamic values and additive offsets
    # in the swapped dimensions. For other dimensions, retain the original
    # mapping.
    new_mapping.input_mapping = {}
    find_iter = lambda x: mapping.input_mapping[dim].find(x)
    for dim in read.memory_type.symbolic_shape:
        dim_iter = [find_iter(x) for x in mapping.iters if find_iter(x)]
        if not dim_iter:
            new_mapping.input_mapping[dim] = mapping.input_mapping[dim]
            continue
        dim_iter = dim_iter[0].pop()
        new_mapping.input_mapping[dim] = mapping.input_mapping[dim].subs(
            {dim_iter: old_to_new_iter[dim_iter]}
        )

    # Update iters for dynamic values in the mapping.
    for dynamic_mapping in new_mapping.dynamic_val_mappings:
        for dim in dynamic_mapping:
            dynamic_mapping[dim] = dynamic_mapping[dim].subs(old_to_new_iter)

    return new_mapping


def construct_shared_read_mapping(read: Read) -> IndexMapping:
    """
    This function constructs a new mapping for the shared read node.
    """
    # Create new identity output mapping.
    mapping = read.mapping
    new_mapping = deepcopy(mapping)
    dynamic_vals = list(mapping.dynamic_val_indices.keys())
    assert mapping.is_output_identity(), "Output mapping must be an identity mapping."
    dim_to_iter = {
        dim: iter for dim, iter in zip(mapping.output_mapping, mapping.iters)
    }

    # Find the iter permutation.
    old_to_new_iter = {}
    for dim in mapping.output_mapping:
        old_to_new_iter[mapping.output_mapping[dim]] = new_mapping.output_mapping[dim]

    for dim, expr in new_mapping.input_mapping.items():
        iter = dim_to_iter[dim]
        new_mapping.input_mapping[dim] = old_to_new_iter[iter]

    new_mapping.dynamic_val_mappings = None
    new_mapping.dynamic_val_indices = {}

    return new_mapping


def update_read_mapping_dynamic_values(read: Read):
    """
    Since every thread is now only reading one element along the gather dimension,
    we need to modify the read of the dynamic value to read only one element
    and do it at the specified index.

    We cannot just update dynamic val node inplace, as it may be used by other
    nodes, clone the node and set the new index and `element_per_thread` on it.

    This code can leave old reads without users, we are expecting them to be
    cleaned up by DCE.
    """
    new_dyn_vals = []
    for dyn_val in read.mapping_dynamic_vals:
        custom = get_custom(dyn_val)
        assert isinstance(custom, Read), f"Only read nodes are supported, got {custom}"
        with custom.graph.inserting_before(dyn_val):
            new_read = Read(
                custom.memory,
                1,
                custom.mapping,
                custom.mapping_dynamic_vals,
            ).add_to_graph(custom.graph)
            new_dyn_vals.append(new_read)

            new_read.index = deepcopy(custom.index)
            for dim in custom.index:
                if dim in read.index:
                    new_read.index[dim] = read.index[dim]

    read.update_arg("mapping_dynamic_vals", new_dyn_vals)


def add_optimized_nodes(
    optimizable_loads: dict[fx.Node, tuple[int, Read]],
    constraint_tile_size: dict[IndexSymbol, int],
    hardware_constraint: HardwareConstraint,
) -> dict[fx.Node, list[fx.Node]]:
    """
    Add optimized global read nodes and shared write nodes to the graph.
    This follows the original implementation in minimize_global_loads, with a couple of changes:
    - We update the mapping of the global reads.
    - We propagate the reads to indirect reads if we have dynamic values.
    - We maintain metadata to update the shared reads in the next function.
    """
    optimized_writes = defaultdict(list)
    shared_read_metadata: dict[Write, SharedReadMetadata] = {}
    for memory, (
        expected_number_of_loads,
        custom_loads,
        _,
        load_elems_per_thread,
        max_elements_per_load,
    ) in optimizable_loads.items():
        original_access_pattern = deepcopy(custom_loads[0].index)
        access_pattern = make_contiguous_index(custom_loads[0])
        for i in range(expected_number_of_loads):
            custom = custom_loads[i]
            with custom.graph.inserting_before(custom.fx_node):
                read = Read(
                    memory,
                    load_elems_per_thread,
                    custom.mapping,
                    custom.mapping_dynamic_vals,
                ).add_to_graph(custom.graph)
                global_offset = (
                    hardware_constraint.linearized_thread_id * load_elems_per_thread
                    + i * max_elements_per_load
                )
                materialized_shape = materialize_shape(
                    constraint_tile_size, custom.memory_type.symbolic_shape
                )
                read.index = construct_min_global_access_pattern(
                    access_pattern,
                    global_offset,
                    load_elems_per_thread,
                    materialized_shape,
                )
                new_custom_read = get_custom(read)
                if new_custom_read.mapping_dynamic_vals:
                    update_read_mapping_dynamic_values(new_custom_read)

                new_custom_read.update_arg(
                    "mapping", construct_global_read_mapping(new_custom_read)
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
                        shared_read_metadata[write] = SharedReadMetadata(
                            index=original_access_pattern,
                            mapping=construct_shared_read_mapping(custom),
                            memory_shape=custom.memory_type.symbolic_shape,
                        )
                        break
    return optimized_writes, shared_read_metadata


def global_to_shared_gathers(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function converts global gathers to shared gathers.
    Unlike minimize_global_loads, this function does not change the widths
    of the loads.
    """
    global_gathers = trace.walk(is_valid_global_gather)
    if not global_gathers:
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
    element_type = get_custom(global_gathers[0]).type.dtype
    load_elems_per_thread = hardware_constraint.max_elems_per_load(element_type)
    max_elements_per_load = total_number_of_threads * load_elems_per_thread

    optimizable_loads = identify_optimizable_loads(
        global_gathers,
        constraint_tile_size,
        load_elems_per_thread,
        max_elements_per_load,
        allow_dynamic_transposed=True,
        use_memory_type=True,
    )

    # Construct new global read nodes and write shared nodes.
    optimized_writes, shared_read_metadata = add_optimized_nodes(
        optimizable_loads,
        constraint_tile_size,
        hardware_constraint,
    )

    # Update all write dependencies.
    update_write_dependencies(optimized_writes, trace, shared_read_metadata)
