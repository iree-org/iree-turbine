# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..._support.tracing import CapturedTrace
from typing import Sequence, Any
from ..constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from torch import fx
from ..._support.indexing import IndexingContext, IndexSymbol
from ...ops.wave_ops import (
    get_custom,
    CustomOp,
    Placeholder,
    IterArg,
    Read,
    Write,
    Reshape,
    NewRegister,
    ReduceOp,
    MMA,
)
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
import itertools
from iree.turbine.kernel._support.dtype import DataType
from ..utils.graph_utils import (
    get_inputs,
)


class ExpansionMetadata:
    def __init__(self, dim_query: dict[IndexSymbol, int] = None):
        self.do_not_expand: bool = False
        self.dim_query = dim_query
        self.last_mma_node = False
        self.source_dim_query = None
        self.num_queries = None
        self.query_index = None

    def __str__(self):
        return str(self.__dict__)


def get_dim_scaling(
    constraints: Sequence[Constraint], node: CustomOp
) -> dict[IndexSymbol, int]:
    """Get the number of expansions for the dimensions based on the constraints for a specific node."""
    dim_scaling: dict[IndexSymbol, int] = {}
    if node.vector_shapes is None:
        return dim_scaling

    hardware_constraints: list[HardwareConstraint] = [
        constraint
        for constraint in constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if len(hardware_constraints) != 1:
        raise ValueError("Exactly one hardware constraint must be provided")

    idxc = IndexingContext.current()
    for constraint in constraints:
        if isinstance(constraint, WorkgroupConstraint) or isinstance(
            constraint, TilingConstraint
        ):
            hw_cons = hardware_constraints[0]
            tile_size = idxc.get_static_value(constraint.tile_size)
            if constraint.dim not in node.vector_shapes:
                continue
            vector_size = node.vector_shapes[constraint.dim]

            # No dim scaling for dims with 0 vector size.
            if vector_size == 0:
                continue

            wave_count = 1
            if isinstance(constraint, WorkgroupConstraint):
                wave_count = hw_cons.waves_per_block[constraint.workgroup_dim]
            if tile_size is None or wave_count is None or vector_size is None:
                raise ValueError(
                    "Tile size, wave count and vector size must be statically known"
                )
            if (
                tile_size % wave_count != 0
                or (tile_size / wave_count) % vector_size != 0
            ):
                raise ValueError(
                    f"Tile size for dim {constraint.dim} must be divisible by "
                    f"wave count and vector size, got: tile_size={tile_size}, "
                    f"wave_count={wave_count}, vector_size={vector_size}"
                )
            dim_scaling[constraint.dim] = tile_size // wave_count // vector_size

    if isinstance(node.type, DataType):
        return {}

    # Also include dimensions that have no constraints on them and are known.
    idxc = IndexingContext.current()
    is_static_dim = lambda dim: dim in idxc.subs
    is_non_batch = lambda dim: node.vector_shapes[dim] > 0
    not_computed = lambda dim: dim not in dim_scaling

    for dim in node.indexing_dims:
        if not_computed(dim) and is_static_dim(dim) and is_non_batch(dim):
            dim_scaling[dim] = idxc.get_static_value(dim) // node.vector_shapes[dim]

    # For reduce ops, also include the reduction dimension.
    if isinstance(node, ReduceOp):
        reduction_dim = node.reduction_dim
        if not_computed(reduction_dim) and is_static_dim(reduction_dim):
            dim_scaling[reduction_dim] = (
                idxc.get_static_value(reduction_dim)
                // node.vector_shapes[reduction_dim]
            )

    return dim_scaling


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def get_indexed_dims(
    all_dims: dict[IndexSymbol, int], nodeOrDims: CustomOp | Sequence[IndexSymbol]
) -> tuple[tuple[IndexSymbol, int], ...]:
    """
    Generates a tuple of (key, value) pairs from the provided dimensions.
    If given a CustomOp instance, it uses its indexing_dims attribute.
    """
    if isinstance(nodeOrDims, CustomOp):
        nodeOrDims = nodeOrDims.indexing_dims
    # Flatten dims for node with multiple values or expanded Reduction.
    if all(isinstance(el, Sequence) for el in nodeOrDims):
        flattened_dims = list(itertools.chain.from_iterable(nodeOrDims))
        flatten_dims_set = dict.fromkeys(flattened_dims)
        nodeOrDims = list(flatten_dims_set)
    return tuple((key, all_dims[key]) for key in nodeOrDims if key in all_dims)


def is_expandable(arg: Any) -> bool:
    """Check if an argument is expandable."""
    if isinstance(arg, Sequence):
        return all(is_expandable(a) for a in arg)
    # Placeholder nodes are only expanded if they are a reduction init arg
    if isinstance(arg, Placeholder) and not isinstance(arg, IterArg):
        return False
    return isinstance(arg, CustomOp)


def get_expanded_name(node: CustomOp, dims: dict[IndexSymbol, int]) -> str:
    """Returns the name of a node with the dimensions appended."""

    node_name = node.fx_node.name
    if isinstance(node, Read) or isinstance(node, Write):
        if get_custom(node.memory).type.address_space == SHARED_ADDRESS_SPACE:
            node_name = node_name + "_shared"
    max_chars = 4
    for key, val in dims.items():
        key_str = str(key)
        if len(key_str) > max_chars:
            key_str = key_str[0:4] + "*"
        node_name += f"_{key_str}:{val}"
    return node_name


def compute_strides(dim_scaling: dict[IndexSymbol, int]) -> list[int]:
    """
    Compute the strides for each dimension based on the dim scaling.
    """
    strides = [1] * len(dim_scaling)
    stride = 1
    for i, dim in enumerate(reversed(dim_scaling.keys())):
        strides[i] = stride
        stride *= dim_scaling[dim]
    return strides[::-1]


def filter_non_cloned_nodes(nodes: list[CustomOp]) -> list[CustomOp]:
    """
    Filter out nodes that have been cloned.
    """
    global expansion_context
    return [node for node in nodes if node not in expansion_context.values()]


def get_reshape_dim_queries(
    reshape: Reshape,
    metadata: ExpansionMetadata,
    dim_scaling: dict[IndexSymbol, int],
    nodes_to_expand: list[tuple[CustomOp, dict[IndexSymbol, int]]],
):
    """
    When expanding a reshape, we have to expand the arguments of the reshape and then concatenate them together
    for the expanded node. Say we have a node with indexing dims = [M, N] with vector shapes m=8, n=2 and
    the reshape wants to map it to m=4, n=4. So we start by expanding the node
    node: {m = 0, n = 0}
        arg: {m = 0, n = 0}
        arg: {m = 0, n = 1}
    node: {m = 1, n = 0}
        arg: {m = 0, n = 0}
        arg: {m = 0, n = 1}
    node: {m = 2, n = 0}
        arg: {m = 1, n = 0}
        arg: {m = 1, n = 1}
    node: {m = 3, n = 0}
        arg: {m = 1, n = 0}
        arg: {m = 1, n = 1}
    ...
    In general,
    For the (m = i, n = j) expansion of the reshape node, we expand the arguments of the reshape node
    using the following recipe:
    - if m_src < m_dst, => we have a one to many mapping from source to destination
        so we expand the arguments along m = i // (m_dst / m_src) and we expand the argument only once.
    - if m_src > m_dst, => we have a many to one mapping from source to destination
        so we expand the arguments along m = i * (m_src / m_dst), ... and we expand the argument m_dst / m_src times.

    In situations where the argument has been expanded along the same dimension, we reuse the expanded node
    by making use of the context.
    """

    dim_combinations = {}
    for dim, value in metadata.dim_query.items():
        if dim not in reshape.target_vector_shape:
            continue
        if reshape.vector_shapes[dim] < reshape.target_vector_shape[dim]:
            scale_factor = (
                reshape.target_vector_shape[dim] // reshape.vector_shapes[dim]
            )
            dim_combinations[dim] = [value // scale_factor]
        else:
            scale_factor = (
                reshape.vector_shapes[dim] // reshape.target_vector_shape[dim]
            )
            begin = value * scale_factor
            dim_combinations[dim] = list(range(begin, begin + scale_factor))
    reshape_dim_combinations = list(itertools.product(*dim_combinations.values()))
    return [
        {dim: val for dim, val in zip(dim_combinations.keys(), combination)}
        for combination in reshape_dim_combinations
    ]


def remove_original_nodes(leaf_nodes: list[CustomOp]):
    """
    Remove the original nodes from the graph.
    """
    queue = leaf_nodes
    while queue:
        custom = queue.pop(0)
        if custom.fx_node._erased:
            continue
        inputs, _ = get_inputs(custom.fx_node, None)
        for input in inputs:
            queue.append(get_custom(input))
        if not custom.users:
            custom.erase()


def remove_unused_registers(trace: CapturedTrace):
    """
    Remove registers that are not used in the graph.
    """
    for node in trace.walk(lambda x: isinstance(get_custom(x), NewRegister)):
        if not node.users:
            node.graph.erase_node(node)
