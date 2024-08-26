from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.tracing import CapturedTrace
from .._support.indexing import IndexingContext, IndexSequence, IndexSymbol, IndexExpr
from ..ops.wave_ops import Read, Write, Output, get_custom
from ..lang.global_symbols import *
from .utils import delinearize_index, DCE
from math import prod
import torch.fx as fx
from collections import defaultdict


def has_write_shared_user(node: Read) -> bool:
    idxc = IndexingContext.current()
    return any(
        isinstance(user, Write)
        and user.type.address_space.subs(idxc.subs) == SHARED_ADDRESS_SPACE
        for user in node.users
    )


def is_valid_global_read(node: fx.Node) -> bool:
    idxc = IndexingContext.current()
    custom = get_custom(node)
    return (
        isinstance(custom, Read)
        and custom.type.address_space.subs(idxc.subs) == GLOBAL_ADDRESS_SPACE
        and has_write_shared_user(custom)
    )


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
    thread_ids = [THREAD_0, THREAD_1, THREAD_2]
    new_index = {key: index[key].subs({t: 0 for t in thread_ids}) for key in index}
    nd_index = delinearize_index(thread_id, shape)
    for i, key in enumerate(index.keys()):
        new_index[key].start += nd_index[i]
        new_index[key].size = load_elems_per_thread
    return new_index


def remove_global_indexing(
    index: dict[IndexSymbol, IndexSequence], tilingConstraints: list[TilingConstraint]
) -> dict[IndexSymbol, IndexSequence]:
    """
    This function takes the index sequence for a global read and removes all
    workgroup and induction level indexing. This is necessary for writes to shared memory
    that operate on promoted memory.
    """
    workgroup_ids = [WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
    new_index = {key: index[key].subs({w: 0 for w in workgroup_ids}) for key in index}
    for key in new_index:
        for constraint in tilingConstraints:
            new_index[key] = new_index[key].subs({constraint.induction_var: 0})
    return new_index


def materialize_shape(
    constraint_tile_size: dict[IndexSymbol, int], symbolic_shape: list[IndexSymbol]
) -> list[int]:
    materialized_shape = []
    idxc = IndexingContext.current()
    for dim in symbolic_shape:
        if dim in constraint_tile_size:
            materialized_shape.append(constraint_tile_size[dim].subs(idxc.subs))
        else:
            materialized_shape.append(dim.subs(idxc.subs))
    return materialized_shape


def identify_optimizable_loads(
    global_read_nodes: list[fx.Node],
    constraint_tile_size: dict[IndexSymbol, int],
    max_elements_per_load: int,
) -> list[Read]:
    """
    Identify sub-optimal global loads that can be removed. A given memory has
    sub-optimal global loads if
        num_global_loads > (M * N) / (T * L)
    where the memory has shape [M, N], there are T threads and each thread can load L elements.
    """
    optimizable_loads: dict[fx.Node, tuple[int, Read]] = {}
    processed_memories = set()
    for read_node in global_read_nodes:
        custom = get_custom(read_node)
        if custom.memory in processed_memories:
            continue
        processed_memories.add(custom.memory)
        materialized_shape = materialize_shape(
            constraint_tile_size, custom.type.symbolic_shape
        )
        # Ensure that the innermost dimension of the shape is a multiple of the elements being loaded.
        if materialized_shape[-1] % max_elements_per_load == 0:
            continue

        total_number_of_elements = prod(materialized_shape)
        expected_number_of_loads = total_number_of_elements // max_elements_per_load
        actual_number_of_loads = len(
            [x for x in global_read_nodes if get_custom(x).memory == custom.memory]
        )
        if expected_number_of_loads >= actual_number_of_loads:
            continue
        optimizable_loads[custom.memory] = (expected_number_of_loads, custom)
    return optimizable_loads


def add_optimized_nodes(
    optimizable_loads: dict[fx.Node, tuple[int, Read]],
    constraint_tile_size: dict[IndexSymbol, int],
    hardware_constraint: HardwareConstraint,
    tilingConstraints: list[TilingConstraint],
    max_elements_per_load: int,
    load_elems_per_thread: int,
) -> list[fx.Node]:
    """
    Add optimized global read nodes and shared write nodes to the graph.
    """
    optimized_writes = defaultdict(list)
    for memory, (expected_number_of_loads, custom) in optimizable_loads.items():
        access_pattern: dict[IndexSymbol, IndexSequence] = custom.index
        for i in range(expected_number_of_loads):
            with custom.graph.inserting_before(custom.fx_node):
                read = Read(memory, load_elems_per_thread).add_to_graph(custom.graph)
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
                for custom_user in custom.users:
                    if (
                        isinstance(custom_user, Write)
                        and custom_user.type.address_space == SHARED_ADDRESS_SPACE
                    ):
                        write = Write(
                            read, custom_user.memory, load_elems_per_thread
                        ).add_to_graph(custom.graph)
                        write.index = remove_global_indexing(
                            read.index, tilingConstraints
                        )
                        optimized_writes[custom_user.memory].append(write)
                        break
    return optimized_writes


def update_write_dependencies(optimized_writes: list[fx.Node], trace: CapturedTrace):
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

        for replaceable_write in trace.walk(is_replaceable_write):
            for user in replaceable_write.users:
                idx = user.args.index([replaceable_write])
                get_custom(user).update_arg(idx, writes)
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
        global_read_nodes, constraint_tile_size, max_elements_per_load
    )

    # Construct new global read nodes and write shared nodes.
    optimized_writes = add_optimized_nodes(
        optimizable_loads,
        constraint_tile_size,
        hardware_constraint,
        [c for c in constraints if isinstance(c, TilingConstraint)],
        max_elements_per_load,
        load_elems_per_thread,
    )

    # Update all write dependencies.
    update_write_dependencies(optimized_writes, trace)
