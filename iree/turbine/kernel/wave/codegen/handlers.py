# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import operator
import sympy
import math
from typing import Any, Callable, ClassVar, Optional, List, Type, Dict, Sequence
import sympy.functions
import sympy.functions.elementary
import sympy.functions.elementary.piecewise
import torch.fx as fx
import torch.utils._pytree as pytree

from ..symbolic_constraints import SymbolicAlias
from ...compiler.ir import (
    Attribute,
    DenseElementsAttr,
    FloatAttr,
    F16Type,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    math_d,
    memref_d,
    stream_d,
    scf_d,
    vector_d,
    llvm_d,
)
from iree.turbine.aot.support.ir_utils import (
    _is_float_type,
    _is_index_type,
    _is_integer_like_type,
    _is_signed_or_signless_type,
    get_conversion_op,
)

# TK infrastructure imports.
from iree.turbine.kernel.lang.global_symbols import *
from ...ops.wave_ops import (
    abs,
    allocate,
    apply_expr,
    broadcast,
    cast,
    conditional,
    eq,
    exp2,
    extract,
    extract_slice,
    ge,
    get_custom,
    get_result,
    gt,
    iterate,
    le,
    log2,
    lt,
    maximum,
    minimum,
    mma,
    permute,
    reciprocal,
    register,
    scalar,
    reshape,
    roundeven,
    scheduling_barrier,
    scheduling_group_barrier,
    self_index,
    select,
    set_symbol,
    shared_memory_barrier,
    shuffle,
    tanh,
    tanh_approx,
)
from ...compiler.base import CodegenError, ValidationError, NDEBUG
from ...compiler.builder import IRProxyValue
from ...compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_py_value,
    cast_vector,
)
from ..constraints import (
    Constraint,
    HardwareConstraint,
    MMAType,
    WorkgroupConstraint,
    TilingConstraint,
)
from ..utils.symbol_utils import subs_idxc
from ..compile_options import WaveCompileOptions

# Indexing imports.
from ..._support.indexing import IndexingContext, IndexExpr, IndexSequence, index_symbol
from ..scheduling.resources import get_scheduling_mask

from .emitter import (
    WaveEmitter,
    handle_op,
    get_type_or_element_type,
    add_emitter_subs,
    gen_sympy_index,
    get_constant_attr,
)


###############################################################################
# Memory Ops
###############################################################################


@handle_op(register)
def handle_register(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    get_thread_shape = lambda index: max(subs_idxc(x.size) for x in index.values())
    shape = [get_thread_shape(get_custom(node).index)]
    vector_shape = cast_py_literal(emitter, shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    vector_type = VectorType.get(vector_shape, element_type)
    register = arith_d.ConstantOp(
        vector_type,
        DenseElementsAttr.get_splat(
            vector_type, get_constant_attr(value, element_type)
        ),
    ).result
    emitter.bind_node_proxy(node, IRProxyValue(register))


@handle_op(scalar)
def handle_scalar(emitter: WaveEmitter, node: fx.Node):
    try:
        value, dtype = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    target_type = IrType.parse(dtype.ir_type_asm())
    i32_type = IntegerType.get_signless(32)
    if isinstance(value, IndexExpr):
        scalar_src = gen_sympy_index(add_emitter_subs(emitter), value)
        scalar_i32 = arith_d.index_cast(i32_type, scalar_src)
        conversion_op = get_conversion_op(
            i32_type, target_type, fastmath=get_fast_math_flags(emitter.options)
        )
        scalar = conversion_op(target_type, scalar_i32)
    elif isinstance(value, (int, float)):
        scalar = arith_d.constant(target_type, value)
    emitter.bind_node_proxy(node, IRProxyValue(scalar))


@handle_op(allocate)
def handle_allocate(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, distributed_shape, dtype, address_space, padding = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    memref_shape = cast_py_literal(emitter, distributed_shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    address_space = Attribute.parse("#gpu.address_space<workgroup>")
    memref_type = MemRefType.get(memref_shape, element_type, None, address_space)
    alloc = memref_d.alloc(memref_type, [], [])
    emitter.bind_node_proxy(node, IRProxyValue(alloc))


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def _build_start_indices(
    emitter: WaveEmitter,
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr],
    dynamic_values: dict[IndexExpr, Any] = {},
) -> list[OpResult]:
    return [
        gen_sympy_index(add_emitter_subs(emitter, dynamic_values), i)
        for i in _get_start_indices(src_indices)
    ]


###############################################################################
# Expressions, Dims and Indexing related ops
###############################################################################


@handle_op(self_index)
def handle_self_index(emitter: WaveEmitter, node: fx.Node):
    try:
        iterator, dtype, elements_per_thread = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    index = get_custom(node).index
    var = index[iterator]
    size = cast_py_literal(emitter, elements_per_thread or subs_idxc(var.size))
    stride = subs_idxc(var.stride)

    start = _get_start_index(var)
    step = IndexingContext.current().iota(size)
    value = start + step * stride
    value = gen_sympy_index(add_emitter_subs(emitter), value)

    element_type = IrType.parse(dtype.ir_type_asm())
    if not isinstance(value.type, VectorType):
        vector_type = VectorType.get([size], value.type)
        value = vector_d.splat(vector_type, value)

    if value.type.element_type != element_type:
        vector_type = VectorType.get([size], element_type)
        value = arith_d.index_cast(vector_type, value)

    emitter.bind_node_proxy(node, IRProxyValue(value))


@handle_op(apply_expr)
def handle_apply_expr(emitter: WaveEmitter, node: fx.Node):
    try:
        args, expr = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    if not isinstance(args, Sequence):
        args = [args]

    symbols = [index_symbol(f"$APPLY_EXPR_ARG_{i}") for i in range(len(args))]
    expr = expr(*symbols)

    index_type = IndexType.get()
    args = [cast_vector(emitter, a, element_type=index_type) for a in args]

    subs = add_emitter_subs(emitter)
    for s, a in zip(symbols, args):
        subs[s] = a

    result = gen_sympy_index(subs, expr)
    emitter.bind_node_proxy(node, IRProxyValue(result))


def _to_scalar(val: Value) -> Value:
    src_type = val.type
    if VectorType.isinstance(src_type):
        assert (src_type.rank == 0 and src_type.shape == []) or (
            src_type.rank == 1 and src_type.shape[0] == 1
        ), f"Only size 0 or 1 vectors are supported: got {src_type}"
        if src_type.rank == 1:
            val = vector_d.extract(val, static_position=[0], dynamic_position=[])

    return val


@handle_op(set_symbol)
def handle_set_symbol(emitter: WaveEmitter, node: fx.Node):
    try:
        symbol, register = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    register = cast_vector(emitter, register, element_type=IndexType.get())
    emitter.dynamic_dims[symbol] = _to_scalar(register)


###############################################################################
# Contraction/MMA Ops
###############################################################################


def emit_mfma(m: int, n: int, k: int, acc: Value, values: list[Value]):
    m = get_constant_attr(m, IntegerType.get_signless(32))
    n = get_constant_attr(n, IntegerType.get_signless(32))
    k = get_constant_attr(k, IntegerType.get_signless(32))
    blocks = get_constant_attr(1, IntegerType.get_signless(32))

    result = amdgpu_d.mfma(
        m=m,
        n=n,
        k=k,
        blocks=blocks,
        source_a=values[0],
        source_b=values[1],
        dest_c=acc,
    )
    return result


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    try:
        lhs, rhs, acc, mma_type = node.args
        acc = cast_vector(emitter, acc)
        values = [cast_vector(emitter, val) for val in [lhs, rhs]]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(acc.type)

    hardware_constraints = [
        constraint
        for constraint in emitter.constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if not hardware_constraints:
        raise CodegenError("No hardware constraints found.")

    m, n, k = hardware_constraints[0].mma_matrix_shapes(mma_type)
    result = emit_mfma(m, n, k, acc, values)
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(shuffle)
def handle_shuffle(emitter: WaveEmitter, node: fx.Node):
    """
    Generate gpu shuffle instruction to enable communication
    between threads in a warp. Currently we only support
    float unit vector that is <= 32 bits.

    Translation to shuffle is done in 3 steps:
    1. Scalarize (vector<1xf16> -> f16)
    2. Pad to 32-bit if needed(f16 -> f32)
    3. Shuffle (gpu.shuffle xor src, offset, width -> f32)
    4. Reconstruct to original vector type (truncf f32 -> f16, broadcast -> vector<1xf16>)

    TODO: Handle non-unit vector types such as vector<4xF8> (useful for resolving layouts).
    """
    try:
        src, offset, width = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    if not isinstance(offset, int) or not isinstance(width, int):
        raise NotImplementedError(
            "Non-const width or offset is not yet implemented for shuffleOp."
        )
    src = cast_py_value(emitter, src).ir_value
    offset = cast_py_value(emitter, offset, IntegerType.get_signless(32)).ir_value
    width = cast_py_value(emitter, width, IntegerType.get_signless(32)).ir_value

    # Shuffle data between other threads in a warp.
    result = gpu_d.shuffle(src, offset, width, gpu_d.ShuffleMode.XOR)[0]

    emitter.bind_node_proxy(node, IRProxyValue(result))


###############################################################################
# Binary math Ops
###############################################################################


def get_rank(mlir_type):
    if not isinstance(mlir_type, ShapedType):
        # Not 0 because vector<f32> is rank 0, and in theory,
        # is broadcastable from pure scalar.
        return -1
    return mlir_type.rank


def handle_binary_op(op):
    def decorator(binary_fn: Callable[[Value, Value], OpResult]):
        @handle_op(op)
        def handle_generic_binary(emitter: WaveEmitter, node: fx.Node):
            try:
                lhs, rhs = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e
            lhs = cast_py_value(emitter, lhs).ir_value
            rhs = cast_py_value(emitter, rhs).ir_value

            # Handle special scalar/rank-0 cases where lhs/rhs may be
            # Dtype, vector<Dtype>, or vector<1xDtype>.
            arg_ranks = [get_rank(arg.type) for arg in (lhs, rhs)]
            if (arg_ranks[0] != arg_ranks[1]) and max(arg_ranks) <= 1:
                if arg_ranks[0] > arg_ranks[1]:
                    # Case where rank(lhs) > rank(rhs)
                    rhs = vector_d.broadcast(lhs.type, rhs)
                else:
                    # Case where rank(rhs) > rank(lhs)
                    lhs = vector_d.broadcast(rhs.type, lhs)

            if lhs.type != rhs.type:
                op = get_custom(node)
                raise ValidationError(
                    f"Expected lhs and rhs to have same type for\n"
                    f"{op}\nGot\n"
                    f"lhs: {lhs.type} vs rhs: {rhs.type}\n"
                    f"{lhs=}\n"
                    f"{rhs=}\n"
                    f"lhs={get_custom(op.lhs)}\n"
                    f"rhs={get_custom(op.rhs)}"
                )

            result = binary_fn(lhs, rhs, emitter.options)

            emitter.bind_node_proxy(node, IRProxyValue(result))

    return decorator


def get_fast_math_flags(options: WaveCompileOptions) -> int | None:
    """
    This function returns the fast math flags for the given options.
    Currently, we turn on all optimizations, but can make this
    more selective.
    """
    return arith_d.FastMathFlags.fast if options.use_fast_math else None


@handle_binary_op(operator.add)
def handle_add(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.addf(lhs, rhs, fastmath=get_fast_math_flags(options))
    elif _is_integer_like_type(element_type):
        result = arith_d.addi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for add: {element_type}")
    return result


@handle_binary_op(operator.sub)
def handle_sub(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.subf(lhs, rhs, fastmath=get_fast_math_flags(options))
    elif _is_integer_like_type(element_type):
        result = arith_d.subi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for sub: {element_type}")
    return result


@handle_binary_op(operator.mul)
def handle_mul(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.mulf(lhs, rhs, fastmath=get_fast_math_flags(options))
    elif _is_integer_like_type(element_type):
        result = arith_d.muli(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for mul: {element_type}")
    return result


@handle_binary_op(operator.truediv)
def handle_div(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.divf(lhs, rhs, fastmath=get_fast_math_flags(options))
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.divsi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for div: {element_type}")
    return result


@handle_binary_op(operator.and_)
def handle_and(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.andi(lhs, rhs)
    else:
        raise ValidationError(
            f"Found unhandled operand type for bitwise and: {element_type}"
        )
    return result


@handle_binary_op(operator.or_)
def handle_or(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.ori(lhs, rhs)
    else:
        raise ValidationError(
            f"Found unhandled operand type for bitwise or: {element_type}"
        )
    return result


@handle_binary_op([operator.gt, gt])
def handle_gt(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.cmpi(arith_d.CmpFPredicate.OGT, lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.cmpi(arith_d.CmpIPredicate.sgt, lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for gt: {element_type}")
    return result


@handle_binary_op([ge, operator.ge])
def handle_ge(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.cmpi(arith_d.CmpFPredicate.OGE, lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.cmpi(arith_d.CmpIPredicate.sge, lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for ge: {element_type}")
    return result


@handle_binary_op([operator.lt, lt])
def handle_lt(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.cmpi(arith_d.CmpFPredicate.OLT, lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.cmpi(arith_d.CmpIPredicate.slt, lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for lt: {element_type}")
    return result


@handle_binary_op([operator.le, le])
def handle_le(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.cmpi(arith_d.CmpFPredicate.OLE, lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.cmpi(arith_d.CmpIPredicate.sle, lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for le: {element_type}")
    return result


@handle_binary_op([operator.eq, eq])
def handle_eq(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.cmpf(arith_d.CmpFPredicate.OEQ, lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.cmpi(arith_d.CmpIPredicate.eq, lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for eq: {element_type}")
    return result


@handle_binary_op(maximum)
def handle_maximum(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.maximumf(lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.maxsi(lhs, rhs)
    else:
        raise ValidationError(
            f"Found unhandled operand type for maximum: {element_type}"
        )
    return result


@handle_binary_op(minimum)
def handle_minimum(lhs: Value, rhs: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.minimumf(lhs, rhs)
    elif _is_integer_like_type(element_type) and _is_signed_or_signless_type(
        element_type
    ):
        result = arith_d.minsi(lhs, rhs)
    else:
        raise ValidationError(
            f"Found unhandled operand type for minimum: {element_type}"
        )
    return result


###############################################################################
# Unary math Ops
###############################################################################


def handle_unary_op(op):
    def decorator(unary_fn: Callable[[Value, Value], OpResult]):
        @handle_op(op)
        def handle_generic_unary(emitter: WaveEmitter, node: fx.Node):
            try:
                (src,) = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e
            src = cast_py_value(emitter, src)

            src = src.ir_value
            result = unary_fn(src, emitter.options)
            emitter.bind_node_proxy(node, IRProxyValue(result))

    return decorator


@handle_unary_op(operator.neg)
def handle_neg(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = arith_d.negf(source, fastmath=get_fast_math_flags(options))
    else:
        raise ValidationError(
            f"Found unhandled operand type for negate: {element_type}"
        )
    return result


@handle_unary_op(exp2)
def handle_exp2(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = math_d.exp2(source)
    else:
        raise ValidationError(f"Found unhandled operand type for exp2: {element_type}")
    return result


@handle_unary_op(log2)
def handle_log2(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = math_d.log2(source)
    else:
        raise ValidationError(f"Found unhandled operand type for exp2: {element_type}")
    return result


@handle_unary_op(reciprocal)
def handle_reciprocal(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        splat_ones = DenseElementsAttr.get_splat(
            source.type, get_constant_attr(1.0, element_type)
        )
        ones = arith_d.ConstantOp(source.type, splat_ones)
        reciprocal = arith_d.divf(ones, source)
    else:
        raise ValidationError(
            f"Found unhandled operand type for reciprocal: {element_type}"
        )
    return reciprocal


@handle_unary_op(abs)
def handle_abs(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        abs = math_d.absf(source)
    elif _is_integer_like_type(element_type):
        abs = math_d.absi(source)
    else:
        raise ValidationError(f"Found unhandled operand type for abs: {element_type}")
    return abs


@handle_unary_op(tanh_approx)
def handle_tanh_approx(source: Value, options: WaveCompileOptions) -> OpResult:
    """
    Reference (cuda C++):
      float a = logits;
      s = fabsf(a);
      t = -2.0f * log2(e) * s;
      e = __builtin_amdgcn_exp2f(t);
      d = e + 1.0f;
      r = __builtin_amdgcn_rcpf(d);
      r = e * (-r) + r;
      if (s < 4.997253418e-3f) r = a;
      r = copysign(r, a);
      logits = params.logits_soft_cap * log2(e) * r;
      return logits;

    Standard definition: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    For x ≥ 0, this is equivalent to: tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))

    To leverage hardware exp2, note that: exp(-2x) = 2^(-2x * log2(e))
    Let: t = -2 * log2(e) * |x|
    Then: e = exp2(t) ≈ exp(-2|x|)
    So, for x ≥ 0: tanh(x) ≈ (1 - e) / (1 + e)

    To avoid division, compute: r0 = reciprocal(1 + e) [using hardware intrinsic] and then: r = (1 - e) * r0
    For very small |x| (below a threshold), we use: tanh(x) ≈ x
    Finally, copy the sign of x
    """

    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        # a = x (source)
        a = source

        # s = |a|
        s = math_d.absf(a)

        # Constants:
        # Compute log2(e)
        log2e_val = math.log2(math.e)

        # Compute factor = -2 * log2(e)
        factor_val = -2.0 * log2e_val

        factor = arith_d.ConstantOp(
            source.type,
            DenseElementsAttr.get_splat(
                source.type, get_constant_attr(factor_val, element_type)
            ),
        )
        # t = factor * s
        t = arith_d.mulf(s, factor, fastmath=get_fast_math_flags(options))

        # e = exp2(t) using fast hardware intrinsic via math_d
        e = math_d.exp2(t)

        one = arith_d.ConstantOp(
            source.type,
            DenseElementsAttr.get_splat(
                source.type, get_constant_attr(1.0, element_type)
            ),
        )

        # d = e + 1.0
        d = arith_d.addf(e, one, fastmath=get_fast_math_flags(options))

        # r = reciprocal(d)
        r = arith_d.divf(one, d, fastmath=get_fast_math_flags(options))

        neg_one = arith_d.ConstantOp(
            source.type,
            DenseElementsAttr.get_splat(
                source.type, get_constant_attr(-1.0, element_type)
            ),
        )

        # r = e * (-r) + r  (computes (e-1)/(e+1) without direct division)
        neg_r = arith_d.mulf(r, neg_one, fastmath=get_fast_math_flags(options))
        temp = arith_d.mulf(e, neg_r, fastmath=get_fast_math_flags(options))
        r = arith_d.addf(temp, r, fastmath=get_fast_math_flags(options))

        # # If s < threshold, then r = a (for small x, tanh(x) ~ x)
        # thresh_val = 4.997253418e-3
        # thresh = arith_d.ConstantOp(
        #     source.type,
        #     DenseElementsAttr.get_splat(source.type, get_constant_attr(thresh_val, element_type))
        # )
        # cmp = arith_d.cmpf(arith_d.CmpFPredicate.OLT, s, thresh)
        # r = arith_d.select(cmp, a, r)

        # Copy the sign of a to r: r = copysign(r, a)
        result = math_d.copysign(r, a)
    else:
        raise ValidationError(f"Unhandled operand type for tanh_approx: {element_type}")
    return result


@handle_unary_op(tanh)
def handle_tanh(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = math_d.tanh(source, fastmath=get_fast_math_flags(options))
    else:
        raise ValidationError(f"Found unhandled operand type for tanh: {element_type}")
    return result


@handle_unary_op(roundeven)
def handle_roundeven(source: Value, options: WaveCompileOptions) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        roundeven = math_d.roundeven(source)
    else:
        raise ValidationError(
            f"Found unhandled operand type for roundeven: {element_type}"
        )
    return roundeven


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(conditional)
def handle_conditional(emitter: WaveEmitter, node: fx.Node):
    try:
        condition, subgraph, implicit_capture = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    if isinstance(condition, sympy.Basic):
        condition = gen_sympy_index(add_emitter_subs(emitter), condition)
    else:
        condition = cast_vector(emitter, condition)
        condition = _to_scalar(condition)

    cond_type = condition.type
    assert IntegerType.isinstance(
        cond_type
    ), f"Condition must me integer, got {cond_type}"

    zero = arith_d.constant(cond_type, 0)
    condition = arith_d.cmpi(arith_d.CmpIPredicate.ne, condition, zero)

    if_op = scf_d.IfOp(condition)
    with InsertionPoint(if_op.then_block) as ip:
        subgraph: fx.Graph = emitter.trace.get_subgraph(subgraph)

        captured_vars: list[fx.Node] = get_custom(node).captured_vars(subgraph)
        for root_v, subgraph_v in zip(implicit_capture, captured_vars):
            emitter._node_values[subgraph_v] = emitter.lookup_node_values(root_v)
        # Emit the subgraph.
        emitter._emit_graph(subgraph)
        scf_d.YieldOp([])


@handle_op(iterate)
def handle_iterate(emitter: WaveEmitter, node: fx.Node):
    try:
        axis, init_args, subgraph, implicit_capture, start, condition = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    if start:
        return handle_iterate_while(emitter, node)

    # Flatten init_args and get IR values for each of them.
    flat_init_args, _ = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    start = arith_d.constant(IndexType.get(), int(0))

    # For now, we assume that dimensions that have tiling constraints on them,
    # do not have any other constraints.
    count = node.count
    if isinstance(count, sympy.Expr):
        end = gen_sympy_index(add_emitter_subs(emitter), count)
    else:
        end = arith_d.constant(IndexType.get(), int(count))

    # Since we divide the end by the tile size, we need to make sure that the
    # step is 1.
    step = arith_d.constant(IndexType.get(), int(1))

    if not step:
        raise CodegenError(
            "Could not determine step size for reduction due to missing tiling constraint."
        )

    forOp = scf_d.ForOp(
        start,
        end,
        step,
        [a.ir_value for a in flat_init_args],
    )
    emitter.induction_vars[axis] = forOp.induction_variable
    with InsertionPoint(forOp.body):
        # Add mapping for iter args.
        subgraph: fx.Graph = emitter.trace.get_subgraph(subgraph)
        iter_args: list[fx.Node] = get_custom(node).iter_args(subgraph)
        assert len(iter_args) == len(forOp.inner_iter_args), (
            f"Len of reduction and for op iter args must match,"
            f" Reduction args: {iter_args};"
            f" For Op args: {[a.type for a in forOp.inner_iter_args]}"
        )
        for i, v in enumerate(forOp.inner_iter_args):
            emitter.bind_node_proxy(iter_args[i], IRProxyValue(v))
        captured_vars: list[fx.Node] = get_custom(node).captured_vars(subgraph)
        for subgraph_v in captured_vars:
            if "lifted" not in subgraph_v.meta:
                raise ValueError(
                    "Cannot find subgraph_v's corresponding value in the root graph."
                )
            root_v = subgraph_v.meta["lifted"]
            emitter._node_values[subgraph_v] = emitter.lookup_node_values(root_v)
        # Emit the subgraph.
        return_values = emitter._emit_graph(subgraph)
        if all(x is None for x in return_values):
            scf_d.YieldOp([])
            return
        # Flattern return values.
        flat_ret_values, _ = pytree.tree_flatten((return_values))
        flat_ret_values = [
            cast_py_value(emitter, value).ir_value for value in flat_ret_values
        ]
        assert len(flat_ret_values) == len(flat_init_args), (
            f"Loop must have the same number of return values as init args, but got\n"
            f"{len(flat_ret_values)} vs {len(flat_init_args)}\n"
            f"{flat_ret_values=}\n"
            f"{flat_init_args=}\n"
        )
        scf_d.YieldOp(flat_ret_values)

    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in forOp.results_])


def handle_iterate_while(emitter: WaveEmitter, node: fx.Node):
    try:
        axis, init_args, subgraph, implicit_capture, start, condition = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Flatten init_args and get IR values for each of them.
    flat_init_args, _ = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]
    flat_init_arg_types = [arg.ir_value.type for arg in flat_init_args]

    # Initialize while loop
    init_value = cast_py_value(emitter, start).ir_value
    if isinstance(init_value.type, VectorType):
        init_value = vector_d.extractelement(init_value, [0])
    if isinstance(init_value.type, IntegerType):
        init_value = arith_d.index_cast(IndexType.get(), init_value)

    assert isinstance(
        init_value.type, IndexType
    ), f"Unhandled start type: {init_value.type}"

    init_args = [arg.ir_value for arg in flat_init_args] + [init_value]
    init_arg_types = flat_init_arg_types + [init_value.type]
    whileOp = scf_d.WhileOp(init_arg_types, init_args)
    whileOp.before.blocks.append(*init_arg_types)
    whileOp.after.blocks.append(*init_arg_types)

    # Before block: condition.
    current_values = whileOp.before.blocks[0].arguments
    with InsertionPoint(whileOp.before.blocks[0]):
        # Replace the axis with a temporary variable when generating the condition
        # to avoid conflicts with the actual value of the axis.
        condition = condition.subs({axis: index_symbol("$TMP")})
        subs = add_emitter_subs(emitter)
        subs[index_symbol("$TMP")] = current_values[-1]
        condition = gen_sympy_index(subs, condition)
        scf_d.ConditionOp(condition, current_values)

    # After block: body.
    current_values = whileOp.after.blocks[0].arguments
    with InsertionPoint(whileOp.after.blocks[0]):
        subgraph = emitter.trace.get_subgraph(subgraph)
        # Map the captured variables from the root graph to the subgraph
        for root_v, subgraph_v in zip(
            implicit_capture, get_custom(node).captured_vars(subgraph)
        ):
            emitter._node_values[subgraph_v] = emitter.lookup_node_values(root_v)

        # Map the iteration variable
        emitter.induction_vars[axis] = current_values[-1]

        # Map the iteration arguments
        iter_args = get_custom(node).iter_args(subgraph)
        for i, arg in enumerate(iter_args):
            if i < len(current_values) - 1:
                emitter.bind_node_proxy(arg, IRProxyValue(current_values[i]))

        # Emit the subgraph
        return_values = emitter._emit_graph(subgraph)
        if all(x is None for x in return_values):
            scf_d.YieldOp([emitter.dynamic_dims[axis]])
            return

        # Flatten return values and yield them
        flat_ret_values, _ = pytree.tree_flatten((return_values))
        flat_ret_values = [
            cast_py_value(emitter, value).ir_value for value in flat_ret_values
        ]
        scf_d.YieldOp(flat_ret_values + [emitter.dynamic_dims[axis]])

    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in whileOp.results_])


###############################################################################
# Synchronization ops
###############################################################################


@handle_op(shared_memory_barrier)
def handle_shared_memory_barrier(emitter: WaveEmitter, node: fx.Node):
    amdgpu_d.lds_barrier()


@handle_op(scheduling_barrier)
def handle_scheduling_barrier(emitter: WaveEmitter, node: fx.Node):
    try:
        operations = node.args[0]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    mask = 0
    for operation in operations:
        mask |= get_scheduling_mask(operation)

    mask = arith_d.constant(IntegerType.get_signless(32), mask)
    llvm_d.call_intrinsic(None, "llvm.amdgcn.sched.barrier", [mask], [], [])


@handle_op(scheduling_group_barrier)
def handle_scheduling_group_barrier(emitter: WaveEmitter, node: fx.Node):
    try:
        instructions, sync_id = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    sync_id = arith_d.constant(IntegerType.get_signless(32), sync_id)
    for instruction, counts in instructions.items():
        mask = get_scheduling_mask(instruction)
        if mask is None:
            continue
        mask = arith_d.constant(IntegerType.get_signless(32), mask)
        counts = arith_d.constant(IntegerType.get_signless(32), counts)
        llvm_d.call_intrinsic(
            None, "llvm.amdgcn.sched.group.barrier", [mask, counts, sync_id], [], []
        )


###############################################################################
# Slicing ops
###############################################################################


@handle_op(extract)
def handle_extract(emitter: WaveEmitter, node: fx.Node):
    try:
        register, offset = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    assert isinstance(offset, list) and len(offset) == 1
    extract_vector = cast_vector(emitter, register)
    result_type = VectorType.get([1], extract_vector.type.element_type)
    element = vector_d.extract_strided_slice(
        result_type,
        extract_vector,
        offset,
        [1],
        [1],
    )

    emitter.bind_node_proxy(node, IRProxyValue(element))


@handle_op(extract_slice)
def handle_extract_slice(emitter: WaveEmitter, node: fx.Node):
    try:
        register, offsets, sizes, strides = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    extract_vector = cast_vector(emitter, register)
    result_type = VectorType.get(sizes, extract_vector.type.element_type)
    element = vector_d.extract_strided_slice(
        result_type,
        extract_vector,
        offsets,
        sizes,
        strides,
    )

    emitter.bind_node_proxy(node, IRProxyValue(element))


###############################################################################
# Reshape ops
###############################################################################


@handle_op(broadcast)
def handle_broadcast(emitter: WaveEmitter, node: fx.Node):
    try:
        register, target_shape = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Get thread_shape/size for broadcast.
    get_thread_shape = lambda index: max(subs_idxc(x.size) for x in index.values())

    src_thread_size = (
        get_thread_shape(register.index)
        if hasattr(register, "index") and register.index
        else None
    )
    target_thread_size = get_thread_shape(node.index) if node.index else None

    # Check MLIR shape
    vector_src = cast_vector(emitter, register)
    vector_type = vector_src.type
    # Only support broadcasting vector<1xdtype> for now.
    if not VectorType.isinstance(vector_type):
        raise NotImplementedError("Scalar src is not implemented yet for shuffleOp.")
    assert (
        vector_type.rank == 0 or vector_type.rank == 1
    ), f"expected vector_type.rank == 1 but got {vector_type}, {node}"

    # Handles scalar broadcast case.
    if vector_type.rank == 0:
        result_type = VectorType.get([target_thread_size], vector_type.element_type)
        element = vector_d.extract(vector_src, static_position=[], dynamic_position=[])
        splat = vector_d.splat(result_type, element)
        emitter.bind_node_proxy(node, IRProxyValue(splat))
        return

    # Handle broadcasting to unit dims as no-op.
    # Most useful for handling broadcast in symbolic shapes.
    src_dims = set(get_custom(register).indexing_dims)
    bcast_dims = list(set(target_shape) - src_dims)
    bcast_sizes = [subs_idxc(node.index[x].size) for x in bcast_dims]
    lane_level_broadcast = target_thread_size != src_thread_size
    if math.prod(bcast_sizes) == 1 and not lane_level_broadcast:
        emitter.bind_node_proxy(node, IRProxyValue(vector_src))
        return

    assert (
        vector_type.shape[0] == 1
    ), f"expected vector_type.shape[0] == 1 but got {vector_type}"

    # Extract and Splat
    # If by chance broadcast size  matches current size, we can return src.
    if target_thread_size == vector_type.shape[0]:
        emitter.bind_node_proxy(node, IRProxyValue(vector_src))
        return

    result_type = VectorType.get([target_thread_size], vector_type.element_type)
    element = vector_d.extract(vector_src, static_position=[0], dynamic_position=[])
    splat = vector_d.splat(result_type, element)
    emitter.bind_node_proxy(node, IRProxyValue(splat))


###############################################################################
# Miscellanous ops
###############################################################################


@handle_op(select)
def handle_select(emitter: WaveEmitter, node: fx.Node):
    try:
        cond, if_true, if_false = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    unwrap = lambda x: cast_py_value(emitter, x).ir_value
    selected = arith_d.select(unwrap(cond), unwrap(if_true), unwrap(if_false))
    emitter.bind_node_proxy(node, IRProxyValue(selected))


@handle_op(get_result)
def handle_get_result(emitter: WaveEmitter, node: fx.Node):
    try:
        value, res_idx = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    for_op = emitter.lookup_node_values(value)[0].owner
    emitter.bind_node_proxy(node, IRProxyValue(for_op.results[res_idx]))


@handle_op(operator.getitem)
def handle_getitem(emitter: WaveEmitter, node: fx.Node):
    if not node.users:
        return
    raise NotImplementedError("getitem: Currently only stub implementation")


def get_float_type(bitwidth: int):
    match bitwidth:
        case 16:
            return F16Type.get()
        case 32:
            return F32Type.get()
        case _:
            raise NotImplementedError(f"Unsupported float bitwidth: {bitwidth}")


@handle_op(cast)
def handle_cast(emitter: WaveEmitter, node: fx.Node):
    try:
        register, dtype = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    vector_src = cast_vector(emitter, register)
    src_vector_type = vector_src.type
    src_elem_type = src_vector_type.element_type
    dst_elem_type = IrType.parse(dtype.ir_type_asm())
    dst_vector_type = VectorType.get(src_vector_type.shape, dst_elem_type)

    if src_vector_type == dst_vector_type:
        emitter.bind_node_proxy(node, IRProxyValue(vector_src))
        return
    conversion_op = get_conversion_op(
        src_elem_type, dst_elem_type, fastmath=get_fast_math_flags(emitter.options)
    )
    casted_vector = conversion_op(dst_vector_type, vector_src)
    emitter.bind_node_proxy(node, IRProxyValue(casted_vector))


@handle_op(permute)
def handle_permute(emitter: WaveEmitter, node: fx.Node):
    try:
        register, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    vector_src = cast_py_value(emitter, register)
    emitter.bind_node_proxy(node, vector_src)


@handle_op(reshape)
def handle_reshape(emitter: WaveEmitter, node: fx.Node):
    try:
        args, target_vector_shapes = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    custom = get_custom(node)

    # Determine whether to extract or combine.
    if len(args) > 1:
        concatenated = None
        for i, sub_arg in enumerate(args):
            vector = cast_vector(emitter, sub_arg)
            shape = vector.type.shape[0]
            if concatenated is None:
                element_type = vector.type.element_type
                vector_type = VectorType.get([shape * len(args)], element_type)
                concatenated = arith_d.ConstantOp(
                    vector_type,
                    DenseElementsAttr.get_splat(
                        vector_type, get_constant_attr(0, element_type)
                    ),
                ).result
            concatenated = vector_d.insert_strided_slice(
                vector, concatenated, [i * shape], [1]
            )
        emitter.bind_node_proxy(node, IRProxyValue(concatenated))
        return

    # Extract the appropriate slice. The offset is obtained from the expanded_dim
    # and so corresponds to the dim_query during expansion. To obtain the
    # actual offset, we need to multiply by the size. The size is obtained by
    # computing the number of partitions using the source and target vector shapes
    # and dividing the incoming vector shape by the number of partitions.
    innermost_dim = custom.type.symbolic_shape[-1]
    offset = custom.expanded_dims[innermost_dim]
    num_partitions = (
        target_vector_shapes[innermost_dim] // custom.vector_shapes[innermost_dim]
    )
    vector = cast_vector(emitter, args[0])
    size = vector.type.shape[0] // num_partitions
    result_type = VectorType.get([size], vector.type.element_type)
    # The offset should only be in [0, num_partitions - 1].
    offset = offset % num_partitions
    slice = vector_d.extract_strided_slice(
        result_type,
        vector,
        [offset * size],
        [size],
        [1],
    )
    emitter.bind_node_proxy(node, IRProxyValue(slice))
