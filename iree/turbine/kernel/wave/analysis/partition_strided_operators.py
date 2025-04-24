# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..constraints import (
    Constraint,
)
from ...lang.global_symbols import *
from ...ops.wave_ops import (
    CustomOp,
    ExtractSlice,
    Read,
    Reshape,
    SelfIndex,
    Write,
    get_custom,
)
from ..._support.indexing import IndexSequence, IndexSymbol
from ..._support.tracing import CapturedTrace
from ....support.logging import get_logger
from ..utils.general_utils import (
    all_equal,
)
from ..utils.mma_utils import (
    simplify_index,
)
from ..utils.symbol_utils import (
    subs_idxc,
)
from copy import deepcopy
from itertools import groupby
from operator import itemgetter
import torch.fx as fx
import numpy as np
import sympy

logger = get_logger("turbine.wave.partition_strided_operators")


def get_concrete_shape(
    vector_shapes_map: dict[IndexSymbol, int],
    symbolic_shape: list[IndexSymbol],
) -> list[int]:
    concrete_shape = [max(vector_shapes_map[dim], 1) for dim in symbolic_shape]
    return concrete_shape


def _get_read_symbolic_shape(
    custom: CustomOp,
):
    return custom.memory_type.symbolic_shape


def _get_write_symbolic_shape(
    custom: CustomOp,
):
    vector_shapes_map = custom.vector_shapes
    memory_shape = custom.memory_type.symbolic_shape
    register_shape = custom.register_type.symbolic_shape
    # Check to see if the memory shape does not match with the vector shapes.
    if not set(memory_shape).issubset(set(vector_shapes_map.keys())):
        return register_shape
    # Pick the shape with the most dimensions.
    if len(memory_shape) > len(register_shape):
        return memory_shape
    return register_shape


def partition_strided_operators(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of operators in the graph
    that are reads or writes on 2d tensors. If the operator has an access pattern where
    the strides are greater than one on a single dimension, this function splits the
    operands into individual elements and constructs a write for
    each individual element.
    """

    op_types = Read | Write
    op_index_fn = {Read: lambda x: x.index, Write: lambda x: x.register_index}

    def has_strided_access(node: fx.Node) -> bool:
        """
        Checks for reads or writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """

        custom = get_custom(node)
        if isinstance(custom, op_types):
            op_type = Read if isinstance(custom, Read) else Write
            index = op_index_fn[op_type](custom)
            strides = [simplify_index(index[dim]).stride for dim in index]
            elements_per_thread = [simplify_index(index[dim]).size for dim in index]
            strides = [x for x, y in zip(strides, elements_per_thread) if y > 1]
            num_strided_accesses = sum(1 for stride in strides if stride > 1)
            if num_strided_accesses > 1:
                raise NotImplementedError(
                    "Support for strided accesses on more than one dimension not implemented yet!"
                )
            return num_strided_accesses == 1
        return False

    strided_operators = trace.walk(has_strided_access)
    for operator in strided_operators:
        custom = get_custom(operator)
        op_type = Read if isinstance(custom, Read) else Write
        index = op_index_fn[op_type](custom)

        simplified_index = {
            dim: simplify_index(index.get(dim, custom.index[dim]))
            for dim in custom.index
        }

        vector_shapes_map = custom.vector_shapes
        symbolic_shape = (
            _get_read_symbolic_shape(custom)
            if op_type == Read
            else _get_write_symbolic_shape(custom)
        )

        concrete_shape = get_concrete_shape(vector_shapes_map, symbolic_shape)
        elements_per_thread = subs_idxc(custom.elements_per_thread)
        max_stride_dim, max_stride = max(
            [(dim, seq.stride) for dim, seq in simplified_index.items()],
            key=lambda item: item[1],
        )

        # Compute offsets we will apply to each index element for each partitioned
        # write.
        offsets = np.array(
            [
                np.unravel_index(int(i * max_stride), concrete_shape)
                for i in range(elements_per_thread)
            ]
        )

        def check_contiguous_index():
            """
            Check if resulted partitioned write is equivalent to contiguous write.

            Write is contiguous if offsets has form `[0,1,2...N]` for
            the fastest mem dim and 0s for all others dims.
            """
            fastest_mem_dim = custom.memory.type.symbolic_shape[-1]

            # Find fastest mem dim index in the register symbolic shape.
            # If we have write with mapping, order of dims between register and
            # mem may differ or mem dims may not even present in reg index.
            if fastest_mem_dim not in symbolic_shape:
                return False

            fastest_mem_dim_idx = symbolic_shape.index(fastest_mem_dim)

            # Construct expected offsets in the form:
            # [[0 0 0]
            #  [0 1 0]
            #  [0 2 0]
            #  [0 3 0]]
            expected_offsets = np.array(
                [
                    [
                        (j if i == fastest_mem_dim_idx else 0)
                        for j in range(elements_per_thread)
                    ]
                    for i in range(len(concrete_shape))
                ]
            ).T
            if not np.array_equal(offsets, expected_offsets):
                return False

            mapping = custom.mapping
            # For writes without mapping this is enough.
            if mapping is None:
                return True

            # If we have mapping, check that fastest dim mapping is trivial, i.e.:
            # IndexMapping(inputs={X: i, ...}, outputs={X: i, ...})
            return (
                mapping.input_mapping.get(fastest_mem_dim, None)
                == mapping.output_mapping[fastest_mem_dim]
            )

        if check_contiguous_index():
            continue

        ops_to_combine = []
        with custom.graph.inserting_before(operator):
            for i in range(elements_per_thread):
                # Non-contiguous access patterns can have varying offsets. We
                # handle that here.
                offset = offsets[i]
                if op_type == Write:
                    extract = ExtractSlice(
                        custom.register_, [i], [1], [1]
                    ).add_to_graph(custom.graph)

                    write = Write(
                        extract,
                        custom.memory,
                        mapping=custom.mapping,
                        elements_per_thread=1,
                    ).add_to_graph(custom.graph)
                    write.index = {
                        dim: IndexSequence(
                            simplified_index[dim].start.subs({GPR_NUM: 0}) + offset[j],
                            1,
                            1,
                        )
                        for j, dim in enumerate(symbolic_shape)
                    }
                    ops_to_combine.append(write)
                else:
                    read = Read(
                        custom.memory,
                        elements_per_thread=1,
                        mapping=custom.mapping,
                        _write_dependency=custom._write_dependency,
                    ).add_to_graph(custom.graph)
                    read.index = {
                        dim: IndexSequence(
                            simplified_index[dim].start.subs({GPR_NUM: 0}) + offset[j],
                            1,
                            1,
                        )
                        for j, dim in enumerate(symbolic_shape)
                    }
                    ops_to_combine.append(read)

        # Update users of original op.
        if isinstance(custom, Write):
            # Useful to handle write/read dependency
            custom.replace_all_uses_with(ops_to_combine)
        elif isinstance(custom, (Read, SelfIndex)):
            with custom.graph.inserting_before(operator):
                reshape = Reshape(ops_to_combine, custom.vector_shapes).add_to_graph(
                    custom.graph
                )
            reshape.expanded_dims = custom.expanded_dims
            reshape.vector_shapes = custom.vector_shapes

            # Save the original index on the reshape op so later we can
            # detect if op was part of `gpr_offset` partition.
            reshape.index = custom.index
            custom.replace_all_uses_with(reshape)

        custom.graph.erase_node(operator)


def partition_ops_with_gpr_offsets(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This function analyzes the index sequence of reads and writes in a graph.
    If the reads or writes have incontiguous offsets based on GPR_NUM, we'd
    need to split these reads/writes appropriately.

    e.g a vector<16xf16> may be owned by lane 0, and lane 16 in this layout:
    [0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 16, 16, 16, 16].

    With our current glossary, this means we have 2 VGPR "chunks".
    [0:4) and [8:12) for lane0, and [4:8) and [12:16) for lane16.
    To the lane it should just look like vector<8xf16>.
    Hence for this example, we'd need two reads of vector<4xf16> and a couple
    insert_slices to combine them together into a single vector<8xf16>.

    """

    def has_gpr_offsets(node: fx.Node) -> bool:
        """
        Checks for writes on 2d tensors with strided access on a single dimension that
        read more than a single element.
        """
        custom = get_custom(node)
        if not isinstance(custom, (Read, Write, SelfIndex)):
            return False
        num_dims_with_gpr = sum(
            1 for v in custom.index.values() if sympy.sympify(v.start).has(GPR_NUM)
        )
        if num_dims_with_gpr == 1:
            return True
        elif num_dims_with_gpr == 0:
            return False
        raise NotImplementedError("Currently only handles 1 dim with GPR offset.")

    strided_operators = trace.walk(has_gpr_offsets)
    for operator in strided_operators:
        custom = get_custom(operator)
        simplified_index = {
            dim: simplify_index(custom.index.get(dim, custom.index[dim]))
            for dim in custom.index
        }
        if isinstance(custom, SelfIndex):
            # If specified use element_per_thread instead of IndexExpr size.
            elements_per_thread = subs_idxc(
                custom.elements_per_thread or custom.index[custom.dim].size
            )
        else:
            elements_per_thread = subs_idxc(custom.elements_per_thread)
        dim_with_gpr_offsets = [
            (k, v.start) for k, v in simplified_index.items() if v.start.has(GPR_NUM)
        ]
        assert len(dim_with_gpr_offsets) == 1, "Expected only 1-Dim has gpr offsets"
        gpr_offset_dim, gpr_offset_expr = dim_with_gpr_offsets[0]
        gpr_offsets = [
            gpr_offset_expr.subs({GPR_NUM: i}) for i in range(elements_per_thread)
        ]

        # Cluster contiguous indices of reads/writes
        # e.g given indices  [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
        # Will cluster to:  [[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24, 25, 26, 27]]
        gpr_relative_offsets = [x - gpr_offsets[0] for x in gpr_offsets]
        gpr_chunks = [
            list(map(itemgetter(1), g))
            for _, g in groupby(enumerate(gpr_relative_offsets), lambda x: x[0] - x[1])
        ]

        # Compute number of GPR chunks.
        num_gpr_chunks = len(gpr_chunks)

        # Compute size of each chunk and ensure they are the same size.
        gpr_sizes = [len(gpr_chunk) for gpr_chunk in gpr_chunks]
        assert all_equal(
            gpr_sizes
        ), "Only support strided GPR offset with uniform sizes."
        gpr_size = gpr_sizes[0]

        # Break apart Reads/Writes that has non-contiguous GPR Read/Writes.
        with custom.graph.inserting_before(operator):
            ops_to_combine = []
            for chunk_id in range(num_gpr_chunks):
                cur_gpr_start_id = chunk_id * gpr_size
                # Get updated index with VGPR offset.
                output_mapping = list(custom.index)
                if hasattr(custom, "mapping") and custom.mapping is not None:
                    output_mapping = list(custom.mapping.output_mapping.keys())
                # Modify stride to 1 S.T we can have vectorized read/write
                # iff gpr_offset_dim is or will be (after mapping) fastest dim.
                updated_index_with_gpr_offset = deepcopy(simplified_index)
                updated_dim_with_gpr_offset = IndexSequence(
                    updated_index_with_gpr_offset[gpr_offset_dim].start.subs(
                        {GPR_NUM: cur_gpr_start_id}
                    ),
                    gpr_size,
                    (
                        1
                        if output_mapping[-1] == gpr_offset_dim
                        else simplified_index[gpr_offset_dim].stride
                    ),
                )
                updated_index_with_gpr_offset[
                    gpr_offset_dim
                ] = updated_dim_with_gpr_offset

                if hasattr(custom, "mapping_dynamic_vals"):
                    # If we are partitioning read/write ops, dynamic_vals can be
                    # potentially partitioned as well. Partitioned dyn vals are
                    # are merged into single value using Reshape op which still
                    # holds the original index containing `GPR_NUM`.
                    # Extract corresponding partitioned chunk from such ops.
                    new_dynamic_vals = []
                    for dyn_val in custom.mapping_dynamic_vals:
                        if any(
                            sympy.sympify(v.start).has(GPR_NUM)
                            for v in get_custom(dyn_val).index.values()
                        ):
                            extract = ExtractSlice(
                                dyn_val, [cur_gpr_start_id], [gpr_size], [1]
                            ).add_to_graph(custom.graph)
                            new_dynamic_vals.append(extract)
                        else:
                            new_dynamic_vals.append(dyn_val)

                # Generate new Read/Write that has contiguous VGPR elements.
                if isinstance(custom, Write):
                    extract = ExtractSlice(
                        custom.register_, [cur_gpr_start_id], [gpr_size], [1]
                    ).add_to_graph(custom.graph)
                    extract.index = updated_index_with_gpr_offset
                    new_node = Write(
                        extract,
                        custom.memory,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        elements_per_thread=gpr_size,
                    ).add_to_graph(custom.graph)
                elif isinstance(custom, Read):
                    # TODO: Add support on how to handle strided reads.
                    new_node = Read(
                        custom.memory,
                        elements_per_thread=gpr_size,
                        mapping=custom.mapping,
                        mapping_dynamic_vals=new_dynamic_vals,
                        _write_dependency=custom._write_dependency,
                    ).add_to_graph(custom.graph)
                elif isinstance(custom, SelfIndex):
                    # iff elements_per_thread is specified, we update
                    # elements_per_thread to chunk size, else return None.
                    self_index_size = gpr_size if custom.elements_per_thread else None
                    new_node = SelfIndex(
                        custom.dim, custom.dtype, self_index_size
                    ).add_to_graph(custom.graph)

                # Update new_node information
                new_node.index = updated_index_with_gpr_offset
                new_node.vector_shapes = custom.vector_shapes
                ops_to_combine.append(new_node)

            # Update users of original op.
            if isinstance(custom, Write):
                # Useful to handle write/read dependency
                custom.replace_all_uses_with(ops_to_combine)
            elif isinstance(custom, (Read, SelfIndex)):
                reshape = Reshape(ops_to_combine, custom.vector_shapes).add_to_graph(
                    custom.graph
                )
                reshape.expanded_dims = custom.expanded_dims
                reshape.vector_shapes = custom.vector_shapes

                # Save the original index on the reshape op so later we can
                # detect if op was part of `gpr_offset` partition.
                reshape.index = custom.index
                custom.replace_all_uses_with(reshape)

            custom.graph.erase_node(custom.fx_node)
