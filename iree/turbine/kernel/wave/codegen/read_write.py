# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
import functools
from typing import Any, Callable, ClassVar, Optional, List, Type, Dict

import torch.fx as fx

from ...compiler.ir import (
    Attribute,
    DenseElementsAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    memref_d,
    scf_d,
    vector_d,
)

from ...compiler.utils import strides_from_symbolic_shape
from ...compiler.builder import IRProxyValue
from ...compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_vector,
)

from ...ops.wave_ops import get_custom, read, write, CustomOp

from ..utils import safe_subs, subs_idxc, find_index_bounds, get_fastest_index

from ..._support.indexing import IndexingContext, IndexExpr, IndexSequence, index_symbol
from ...lang.wave_types import IndexMapping
from ...lang.global_symbols import *

from .emitter import (
    WaveEmitter,
    handle_op,
    get_type_or_element_type,
    add_emitter_subs,
    gen_sympy_index,
    get_constant_attr,
)


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def _split_index(src: IndexExpr | int) -> tuple[IndexExpr, IndexExpr]:
    """
    Split index expr into thread-dependent and thread-independent parts
    """
    subs_wg = {WORKGROUP_0: 0, WORKGROUP_1: 0, WORKGROUP_2: 0}
    # Replace all wg symbols with 0s to get thread-dependent index.
    # All dynamic values will also be part of thread-index.
    thread_dependent_index = safe_subs(src, subs_wg)

    # Compute thread-independent index as `orig_index - thread_dependent_index`
    # All thread symbols and dynamic should cancel-out in the result.
    thread_independent_index = sympy.simplify(src - thread_dependent_index)
    if thread_independent_index.free_symbols - set(subs_wg.keys()):
        # If we have any symbols besides wg symbols, means some thread or
        # dynamic symbols were not canceled out, use the entire index as
        # thread dependent index.
        thread_independent_index = sympy.sympify(0)
        thread_dependent_index = src

    return thread_independent_index, thread_dependent_index


def _build_start_indices(
    emitter: WaveEmitter,
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
    dynamic_values: dict[IndexExpr, Any] = {},
) -> tuple[list[OpResult], list[OpResult], list[OpResult]]:
    start_indices = _get_start_indices(src_indices)
    split_indices = [_split_index(i) for i in start_indices]
    subs = add_emitter_subs(emitter, dynamic_values)
    indices = [gen_sympy_index(subs, i) for i in start_indices]
    indices_wg = [gen_sympy_index(subs, i[0]) for i in split_indices]
    indices_th = [gen_sympy_index(subs, i[1]) for i in split_indices]

    return indices, indices_wg, indices_th


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def _get_symbolic_shape(node: fx.Node) -> tuple[IndexExpr]:
    return get_custom(node).type.symbolic_shape


def _build_mask(
    emitter: WaveEmitter, index: Dict[IndexExpr, IndexExpr], elements_per_thread: int
) -> Optional[OpResult]:
    bounds = find_index_bounds(emitter.constraints, index)
    if bounds is None:
        return None

    idxc = IndexingContext.current()
    fastest_dim = get_fastest_index(index)
    last_dim = list(index)[fastest_dim]
    new_index = {k: _get_start_index(v) for k, v in index.items()}

    new_index[last_dim] = new_index[last_dim] + idxc.iota(elements_per_thread)

    mask_expr = functools.reduce(
        lambda a, b: sympy.And(a, b), (new_index[dim] < dim for dim in bounds)
    )
    mask = gen_sympy_index(add_emitter_subs(emitter), mask_expr)

    mask_vec_type = VectorType.get([elements_per_thread], IntegerType.get_signless(1))
    if mask.type != mask_vec_type:
        mask = vector_d.splat(mask_vec_type, mask)

    return mask


def _get_splat_const(vec_type: IrType, value: Any) -> Value:
    splat = DenseElementsAttr.get_splat(
        vec_type, get_constant_attr(value, vec_type.element_type)
    )
    return arith_d.constant(vec_type, splat)


def _constant_mask(vec_type: IrType) -> Value:
    return _get_splat_const(vec_type, 1)


def _construct_gather_scatter_indices(
    emitter: WaveEmitter,
    symbolic_shape: tuple[IndexExpr],
    index: tuple[IndexExpr],
    mapping: IndexMapping,
    elements_per_thread: int,
    is_read: bool,
    dynamic_vals: tuple[Any, ...],
    is_contiguous: bool,
    memory: CustomOp,
) -> tuple[list[OpResult], list[OpResult], list[OpResult], OpResult, OpResult]:
    # Apply symbolic_shape order to indices, e.g. if original mapping is
    # {M: iter(0), N: iter(1)} and symbolic_shape is (N, M), result will
    # be (iter(1), iter(0))
    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        index_mapping = mapping.map_input_indices(symbolic_shape)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolic_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(i.subs(idxc.subs) for i in index_mapping)

    iters = mapping.iters

    # As we only support identity input/output mapping for now, we can directly
    # substitute iterators with corresponding expanded index.
    subs = [
        (sym, expr.start) for sym, expr in zip(iters.keys(), index.values())
    ] + list(idxc.subs.items())

    # Contruct input/output index, substituting iterators in input mapping with
    # expanded index.
    result_index = {key: m.subs(subs) for key, m in zip(symbolic_shape, index_mapping)}

    mask = _build_mask(emitter, index, elements_per_thread)
    if mask is None:
        mask_vec_type = VectorType.get(
            [elements_per_thread], IntegerType.get_signless(1)
        )
        mask = _constant_mask(mask_vec_type)

    def extract0(src):
        static_pos = [0] * src.type.rank
        return vector_d.extract(src, static_position=static_pos, dynamic_position=[])

    dynamic_vals_map_start = {
        sym: extract0(val)
        for sym, val in zip(mapping.dynamic_val_indices.keys(), dynamic_vals)
    }
    if is_contiguous:
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        return start_indices, start_indices_wg, start_indices_th, None, mask

    start_indices = _get_start_indices(result_index)
    start_indices_orig = _get_start_indices(index)
    fastest_dim = get_fastest_index(index)
    need_dynamic_offsets = False
    for val in dynamic_vals:
        shape = val.type.shape
        assert shape in (
            [1],
            [elements_per_thread],
        ), f"Dynamic val shape must be {[1]} or {[elements_per_thread]} but got {shape}"
        if shape[0] > 1:
            need_dynamic_offsets = True

    offsets = []
    if memory.type.address_space == SHARED_ADDRESS_SPACE:
        symbolic_shape = memory.distributed_shape
    strides = strides_from_symbolic_shape(idxc, symbolic_shape, allow_mixed_shapes=True)
    start_indices_offset = _compute_offset(start_indices, strides)
    for i in range(elements_per_thread):
        # Update fastest dim, i.e. in case of identity mapping it will
        # be equivalent to just vector.load
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        subs[fastest_dim] = (subs[fastest_dim][0], start_indices_orig[fastest_dim] + i)
        indices = [i.subs(subs) for i in index_mapping]

        # First, we build indices as if resulting gather/scatter `start_indices`
        # are 0 as mapping expression may depend on absolute value of index
        # (e.g. `index % 32`). Then we adjust for the non-0 `start_indices` by
        # subtracting computed previously linear `start_indices_offset`. For
        # simple cases like transpose, the resulting expression should fold into
        # simple constant while more complex expressions may requires actual
        # arith ops on dynamic values.
        offset = _compute_offset(indices, strides) - start_indices_offset
        offset = subs_idxc(offset)

        if offset.is_number:
            # If resulted offset sympy expr is convertible to int constant it
            # will be directly encoded into `arith.constant`.
            # For non-constant expressions, we will generate a real sequence of
            # arith ops and then `vector.insertelement` them into offsets vec.
            offset = int(offset)
        else:
            need_dynamic_offsets = True
            break

        offsets.append(offset)

    offsets_vec_type = VectorType.get([elements_per_thread], IndexType.get())
    if need_dynamic_offsets:
        # In case we need dynamic `offsets_vec`, set all `start_indices` to 0
        # and encode entire index info in `offsets_vec`.
        result_index = {key: 0 for key in symbolic_shape}
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        # Last item in `subs` corresponds to last item in `start_indices_orig`
        # which is fastest changing dim.
        # Replacing last element with `idxc.iota(elements_per_thread)` will
        # generate vectorized index code, each element in it corresponding to
        # individual vector element index.
        subs[-1] = (
            subs[-1][0],
            start_indices_orig[-1] + idxc.iota(elements_per_thread),
        )
        dynamic_vals_map = {
            sym: val
            for sym, val in zip(mapping.dynamic_val_indices.keys(), dynamic_vals)
        }
        indices = [i.subs(subs) for i in index_mapping]
        offsets_vec = gen_sympy_index(
            add_emitter_subs(emitter, dynamic_vals_map),
            _compute_offset(indices, strides),
        )
    else:
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, result_index, dynamic_vals_map_start
        )
        if offsets == list(range(elements_per_thread)):
            return start_indices, start_indices_wg, start_indices_th, None, mask

        offsets = [IntegerAttr.get(IndexType.get(), off) for off in offsets]
        offsets_vec = arith_d.ConstantOp(
            offsets_vec_type, DenseElementsAttr.get(offsets, offsets_vec_type)
        )

    return start_indices, start_indices_wg, start_indices_th, offsets_vec, mask


def _get_max_buffer_size(elem_type: IrType) -> int:
    """
    Return max memref size suitable for buffer ops.

    Buffer ops offsets are i32, return maximum memref size in elements.
    """
    return ((1 << 31) - 1) // (elem_type.width // 8)


def _linearize_memref(
    mem: Value,
    offsets_wg: tuple[Value | int],
    offsets_th: tuple[Value | int],
    strides: tuple[Value],
) -> tuple[Value, Value]:
    """
    Convert n-D memref into 1-D memref, suitable for buffer ops.

    Apply offsets to the memref and convert result to 1-D. Resulting memref size
    is set to `max_buffer_size - 1` so buffer access to the last element will be
    no-op.
    """
    memref_type = mem.type
    offset = None
    offset_th = None
    overflow_flags = arith_d.IntegerOverflowFlags.nsw
    for ind_wg, ind_th, stride in zip(offsets_wg, offsets_th, strides):
        if isinstance(ind_wg, int):
            ind_wg = arith_d.constant(IndexType.get(), ind_wg)

        if isinstance(ind_th, int):
            ind_th = arith_d.constant(IndexType.get(), ind_th)

        off_wg = arith_d.muli(ind_wg, stride, overflow_flags=overflow_flags)
        if offset is None:
            offset = off_wg
        else:
            offset = arith_d.addi(offset, off_wg, overflow_flags=overflow_flags)

        off_th = arith_d.muli(ind_th, stride, overflow_flags=overflow_flags)
        if offset_th is None:
            offset_th = off_th
        else:
            offset_th = arith_d.addi(offset_th, off_th, overflow_flags=overflow_flags)

    size_full = arith_d.constant(
        IndexType.get(), _get_max_buffer_size(memref_type.element_type) - 1
    )

    dyn_val = ShapedType.get_dynamic_size()
    res_shape = [dyn_val]
    element_type = memref_type.element_type
    memory_space = memref_type.memory_space
    resut_type = MemRefType.get(
        res_shape,
        element_type,
        layout=Attribute.parse("strided<[1], offset: ?>"),
        memory_space=memory_space,
    )
    return (
        memref_d.reinterpret_cast(
            resut_type,
            mem,
            offsets=[offset],
            sizes=[size_full],
            strides=[],
            static_offsets=[dyn_val],
            static_sizes=[dyn_val],
            static_strides=[1],
        ),
        offset_th,
    )


def _get_splat_input(src: Optional[Value]) -> Optional[Value]:
    """
    If `src` is vector.splat result, return splat input, otherwise return None.
    """
    if src is None:
        return None

    owner = getattr(src, "owner", None)
    if owner is None:
        return None

    op = src.owner.opview
    if isinstance(op, vector_d.SplatOp):
        return op.input

    return None


def _create_buffer_read_write(
    elem_type: IrType, ptr: Value, offset: Value, value: Optional[Value] = None
) -> Optional[Value]:
    # Buffer ops doesn't support 1-element vectors, convert to scalar.
    is_1elem = isinstance(elem_type, VectorType) and elem_type.shape == [1]
    if value is None:
        load_type = elem_type
        if is_1elem:
            load_type = elem_type.element_type

        res = amdgpu_d.raw_buffer_load(load_type, ptr, indices=[offset])
        if is_1elem:
            res = vector_d.splat(elem_type, res)

        return res
    else:
        if is_1elem:
            value = vector_d.extract(value, static_position=[0], dynamic_position=[])
        amdgpu_d.raw_buffer_store(value, ptr, indices=[offset])
        return None


def _create_vec_read_write(
    emitter: WaveEmitter,
    symbolic_shape: tuple[IndexExpr, ...],
    mem: Value,
    value: Optional[Value],
    vector_type: Optional[IrType],
    start_indices: tuple[Value],
    start_indices_wg: tuple[Value],
    start_indices_th: tuple[Value],
    elements_per_thread: int,
    memory: CustomOp,
    mask: Optional[Value],
    offsets_vec: Optional[Value],
) -> Optional[Value]:
    is_read = value is None
    if mask is None and offsets_vec is None:
        if is_read:
            return vector_d.load(vector_type, mem, start_indices)
        else:
            vector_d.store(value, mem, start_indices)
            return

    mask_splat = _get_splat_input(mask)
    splatted_masked = offsets_vec is None and mask_splat is not None

    # Only use buffer ops if it's gather/scatter or splated masked op on global mem.
    use_buffer_ops = (
        offsets_vec is not None or splatted_masked
    ) and mem.type.memory_space is None

    if vector_type is None:
        vector_type = value.type

    element_type = vector_type.element_type

    if is_read:
        zero = get_constant_attr(0, element_type)
        zero = arith_d.constant(element_type, zero)

    if memory.type.address_space == SHARED_ADDRESS_SPACE:
        symbolic_shape = memory.distributed_shape
    strides = strides_from_symbolic_shape(
        IndexingContext.current(), symbolic_shape, allow_mixed_shapes=True
    )

    def extract(vec, ind):
        return vector_d.extract(vec, static_position=[ind], dynamic_position=[])

    # TODO: If strides cannot be converted into integers, means they are dynamic
    # and linearize breaks, need to investigate later.
    has_int_strides = all(isinstance(s, int) for s in strides)
    optname = "use_buffer_load_ops" if is_read else "use_buffer_store_ops"
    buffer_ops_enabled = emitter.params.get(optname, False)
    if buffer_ops_enabled and has_int_strides and use_buffer_ops:
        strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in strides]
        data, offset_th = _linearize_memref(
            mem, start_indices_wg, start_indices_th, strides
        )

        if offsets_vec is None:
            offsets_vec_type = VectorType.get(vector_type.shape, IndexType.get())
            vals = [
                IntegerAttr.get(IndexType.get(), v) for v in range(elements_per_thread)
            ]
            offsets_vec = arith_d.constant(
                offsets_vec_type, DenseElementsAttr.get(vals, offsets_vec_type)
            )

        if splatted_masked:
            # If mask value is same for all vector elements, we can use vector
            # buffer ops.
            i32 = IntegerType.get_signless(32)
            offset_th = arith_d.index_cast(i32, offset_th)
            oob_idx = _get_max_buffer_size(element_type)
            oob_idx = arith_d.constant(i32, oob_idx)
            offset_th = arith_d.select(mask_splat, offset_th, oob_idx)

            if is_read:
                return _create_buffer_read_write(vector_type, data, offset_th)
            else:
                _create_buffer_read_write(vector_type, data, offset_th, value)
                return
        else:
            # If mask value is different for each element, unroll op to
            # individual values.
            offset_th = vector_d.splat(offsets_vec.type, offset_th)
            offsets_vec = arith_d.addi(offsets_vec, offset_th)
            if mask is not None:
                i32 = IntegerType.get_signless(32)
                i32vec = VectorType.get([elements_per_thread], i32)
                offsets_vec = arith_d.index_cast(i32vec, offsets_vec)
                oob_idx = _get_max_buffer_size(element_type)
                oob_idx = arith_d.constant(i32, oob_idx)
                oob_idx = vector_d.splat(offsets_vec.type, oob_idx)
                offsets_vec = arith_d.select(mask, offsets_vec, oob_idx)

            if is_read:
                elements = []
                for i in range(elements_per_thread):
                    offset = extract(offsets_vec, i)

                    if mask is None:
                        elem = memref_d.load(element_type, data, indices=[offset])
                    else:
                        elem = _create_buffer_read_write(element_type, data, offset)

                    elements.append(elem)

                return vector_d.from_elements(vector_type, elements)
            else:
                for i in range(elements_per_thread):
                    offset = extract(offsets_vec, i)

                    elem = extract(value, i)

                    if mask is None:
                        memref_d.store(elem, data, indices=[offset])
                    else:
                        _create_buffer_read_write(vector_type, data, offset, elem)

                return

    else:

        if offsets_vec is None:
            if is_read:
                passthru = vector_d.splat(vector_type, zero)
                return vector_d.maskedload(
                    vector_type, mem, start_indices, mask, passthru
                )
            else:
                vector_d.maskedstore(mem, start_indices, mask, value)
                return

        if mask is None:
            mask_vec_type = VectorType.get(
                [elements_per_thread], IntegerType.get_signless(1)
            )
            mask = _constant_mask(mask_vec_type)

        # TODO: Need static strides for linearize to work.
        if has_int_strides:
            vec1 = VectorType.get([1], element_type)
            vec1_mask = VectorType.get([1], IntegerType.get_signless(1))
            strides = [gen_sympy_index(add_emitter_subs(emitter), s) for s in strides]
            data, _ = _linearize_memref(
                mem, start_indices, (0,) * len(start_indices), strides
            )

            # Unroll gather/scatter into individual masked ops.
            # Vector canonicalizations will convert them into unmasked later if
            # mask is constant.
            if is_read:
                passthru = vector_d.splat(vec1, zero)
                elements = []
                for i in range(elements_per_thread):
                    mask_elem = extract(mask, i)
                    mask_elem = vector_d.splat(vec1_mask, mask_elem)

                    offset = extract(offsets_vec, i)

                    elem = vector_d.maskedload(
                        vec1, data, [offset], mask_elem, passthru
                    )
                    elements.append(elem)

                elements = [extract(v, 0) for v in elements]
                return vector_d.from_elements(vector_type, elements)
            else:
                for i in range(elements_per_thread):
                    mask_elem = extract(mask, i)
                    mask_elem = vector_d.splat(vec1_mask, mask_elem)

                    offset = extract(offsets_vec, i)

                    elem = extract(value, i)
                    elem = vector_d.splat(vec1, elem)

                    vector_d.maskedstore(data, [offset], mask_elem, elem)

                return

        if is_read:
            passthru = vector_d.splat(vector_type, zero)
            return vector_d.gather(
                vector_type, mem, start_indices, offsets_vec, mask, passthru
            )
        else:
            vector_d.scatter(mem, start_indices, offsets_vec, mask, value)
            return


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping, dyn_vals, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    input_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    if get_custom(node).has_identity_mapping():
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, index
        )
        mask = _build_mask(
            emitter,
            index,
            elements_per_thread,
        )
        result = _create_vec_read_write(
            emitter,
            input_shape,
            kb_src,
            None,
            vector_type,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            get_custom(memory),
            mask,
            offsets_vec=None,
        )
    else:
        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
        )
        (
            start_indices,
            start_indices_wg,
            start_indices_th,
            offsets_vec,
            mask,
        ) = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolic_shape=input_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=True,
            dynamic_vals=dyn_vals,
            is_contiguous=get_custom(node).is_contiguous_vec(),
            memory=get_custom(memory),
        )
        result = _create_vec_read_write(
            emitter,
            input_shape,
            kb_src,
            None,
            vector_type,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            get_custom(memory),
            mask,
            offsets_vec,
        )

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        register, memory, elements_per_thread, mapping, dyn_vals = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # memory has no IR node yet.
    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    vector_shape = cast_py_literal(emitter, (elements_per_thread,))

    # TODO: Support elements_per_thread size mismatch and broadcasting

    assert (
        tuple(insert_type.shape) == vector_shape
    ), f"Shape doesn't match: {tuple(insert_type.shape)} and {(vector_shape)}"

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected write to have index attr.")

    index = node.index

    input_shape = _get_symbolic_shape(register)
    output_shape = _get_symbolic_shape(memory)
    elements_per_thread = cast_py_literal(emitter, elements_per_thread)
    if get_custom(node).has_identity_mapping():
        start_indices, start_indices_wg, start_indices_th = _build_start_indices(
            emitter, index
        )
        mask = _build_mask(emitter, index, elements_per_thread)
        _create_vec_read_write(
            emitter,
            output_shape,
            kb_dest,
            insert_vector,
            None,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            get_custom(memory),
            mask,
            offsets_vec=None,
        )
    else:
        assert (
            input_shape == mapping.input_shape
        ), "non-identity input mapping is not supported yet"

        dyn_vals = tuple(
            cast_vector(emitter, reg, element_type=IndexType.get()) for reg in dyn_vals
        )
        (
            start_indices,
            start_indices_wg,
            start_indices_th,
            offsets_vec,
            mask,
        ) = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolic_shape=output_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=elements_per_thread,
            is_read=False,
            dynamic_vals=dyn_vals,
            is_contiguous=get_custom(node).is_contiguous_vec(),
            memory=get_custom(memory),
        )

        _create_vec_read_write(
            emitter,
            output_shape,
            kb_dest,
            insert_vector,
            None,
            start_indices,
            start_indices_wg,
            start_indices_th,
            elements_per_thread,
            get_custom(memory),
            mask,
            offsets_vec,
        )
