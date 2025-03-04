# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
from typing import Any, Callable, ClassVar, Optional, List, Type, Dict
from dataclasses import dataclass
from collections import namedtuple

import torch.fx as fx

from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.aot.support.ir_utils import (
    _is_float_type,
    _is_index_type,
    _is_integer_like_type,
)


from ...compiler.ir import (
    Attribute,
    DenseElementsAttr,
    FloatAttr,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    arith_d,
    func_d,
    gpu_d,
    stream_d,
    vector_d,
)


from ...compiler.builder import IRProxyValue
from ...compiler.kernel_codegen import BoundKernelSignature
from ..._support.tracing import CapturedTrace
from ...compiler.base import CodegenError, NDEBUG

from ...lang.wave_types import IndexSymbol
from ..constraints import Constraint, TilingConstraint
from ..._support.indexing import IndexingContext, IndexExpr


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    constraints: list[Constraint]
    dynamic_symbols: list[IndexSymbol]
    params: dict[str, Any]
    ip: InsertionPoint = None
    OP_HANDLERS: ClassVar[dict[str, Callable[["WaveEmitter", fx.Node], None]]] = {}
    _node_values: ClassVar[dict[fx.Node, List[IRProxyValue]]] = {}

    def __post_init__(self):
        self.ip = InsertionPoint(self.root_sig.entry_block)

    def emit_program_invariants(self):
        self.workgroup_ids = [
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 0)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 1)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 2)),
        ]
        self.thread_ids = [
            gpu_d.thread_id(gpu_d.Dimension.x),
            gpu_d.thread_id(gpu_d.Dimension.y),
            gpu_d.thread_id(gpu_d.Dimension.z),
        ]
        self.induction_vars: dict[IndexSymbol, Value] = {}
        self.dynamic_dims: dict[IndexSymbol, Value] = {}
        symbol_iterator = iter(self.dynamic_symbols)
        for arg in self.root_sig.entry_block.arguments:
            if arg.type == IndexType.get():
                self.dynamic_dims[next(symbol_iterator)] = arg

    def emit(self, graph: Optional[fx.Graph] = None):
        with self.ip, Location.unknown():
            self.emit_program_invariants()
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])

    def _emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self._emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def _emit_function_call_node(self, node: fx.Node):
        target_op = node.tkw_op_name
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")

        handler(self, node)

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self.root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        values = [v.ir_value if isinstance(v, IRProxyValue) else v for v in values]
        return values

    def bind_node_proxy(self, node: fx.Node, proxy: IRProxyValue):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        self._node_values[node] = [proxy]

    def bind_node_proxies(self, node: fx.Node, proxies: List[IRProxyValue]):
        assert NDEBUG or (
            isinstance(node, fx.Node)
            and all(isinstance(p, IRProxyValue) for p in proxies)
        )
        self._node_values[node] = proxies

    def get_induction_vars_and_syms(self) -> tuple[list[OpResult], list[IndexExpr]]:
        induction_var_syms = []
        induction_vars = []
        if self.induction_vars:
            for constraint in self.constraints:
                if isinstance(constraint, TilingConstraint):
                    assert (
                        constraint.dim in self.induction_vars
                    ), f"Could not find induction var for {constraint.dim} dimension"
                    induction_var_syms.append(constraint.induction_var)
                    induction_vars.append(self.induction_vars[constraint.dim])

        return induction_vars, induction_var_syms


def handle_op(op: Callable[..., Any] | list[Callable[..., Any]]):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None]
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        if isinstance(op, Callable):
            WaveEmitter.OP_HANDLERS[op.__name__] = f
        elif isinstance(op, list):
            for op_iter in op:
                WaveEmitter.OP_HANDLERS[op_iter.__name__] = f
        else:
            raise ValueError("handle_op only handle Callable or list of Callable")
        return f

    return decorator


def get_type_or_element_type(operand_type: IrType):
    assert isinstance(operand_type, IrType)
    if isinstance(operand_type, ShapedType):
        return operand_type.element_type
    else:
        return operand_type


def add_emitter_subs(
    emitter: WaveEmitter, dynamic_values: dict[IndexExpr, Value] = {}
) -> dict[IndexSymbol, Value]:
    induction_vars, induction_var_syms = emitter.get_induction_vars_and_syms()

    # TODO: factor this out
    all_symbols = emitter.thread_ids + emitter.workgroup_ids + induction_vars
    dynamics = dict(
        zip(
            [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
            + induction_var_syms,
            all_symbols,
        )
    )
    dynamics.update(dynamic_values)
    dynamics.update(emitter.dynamic_dims)
    return dynamics


_emulate_ceildiv = True

_Rational = namedtuple("_Rational", ["numerator", "denominator"])


def gen_sympy_index(dynamics: dict[IndexSymbol, Value], expr: sympy.Expr) -> OpResult:
    stack: list[OpResult] = []

    def _get_ir_value(arg):
        if not isinstance(arg, (Value, OpResult)):
            arg = arg.result

        return arg

    def _check_vec_scalar(a, b):
        if not isinstance(a.type, VectorType):
            return False

        if a.type.element_type == b.type:
            return True

        return (
            isinstance(b.type, VectorType)
            and b.type.shape == [1]
            and a.type.element_type == b.type.element_type
        )

    def _broadcast(a, b):
        a = _get_ir_value(a)
        b = _get_ir_value(b)

        if a.type == b.type:
            return a, b

        if _check_vec_scalar(a, b):
            if isinstance(b.type, VectorType):
                b = vector_d.extract(b, static_position=[0], dynamic_position=[])

            b = vector_d.splat(a.type, b)
            return a, b

        if _check_vec_scalar(b, a):
            if isinstance(a.type, VectorType):
                a = vector_d.extract(a, static_position=[0], dynamic_position=[])

            a = vector_d.splat(b.type, a)
            return a, b

        raise CodegenError(f"Cannot broadcast {a.type} and {b.type}")

    def get_const_val(arg):
        if isinstance(arg, OpResult):
            arg = arg.owner.opview

        if isinstance(arg, arith_d.ConstantOp):
            value = arg.attributes["value"]
            if isinstance(value, IntegerAttr):
                return int(value)

        return None

    overflow_flags = arith_d.IntegerOverflowFlags.nsw | arith_d.IntegerOverflowFlags.nuw

    def muli(lhs, rhs):
        if get_const_val(lhs) == 1:
            return rhs

        if get_const_val(rhs) == 1:
            return lhs

        return arith_d.muli(lhs, rhs, overflow_flags=overflow_flags)

    def addi(lhs, rhs):
        if get_const_val(lhs) == 0:
            return rhs

        if get_const_val(rhs) == 0:
            return lhs

        if (rhs_val := get_const_val(rhs)) and rhs_val < 0:
            # If rhs_val is negative, since it is of index type and we have
            # overflow flags, we need to express it as a sub to get the
            # correct result.
            return arith_d.subi(
                lhs, _get_const(-rhs_val), overflow_flags=overflow_flags
            )

        return arith_d.addi(lhs, rhs, overflow_flags=overflow_flags)

    # `x + (a/b)` transformed into `(x*b + a) / b`
    def _add(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli(*_broadcast(lhs.denominator, rhs))
            numerator = addi(*_broadcast(numerator, lhs.numerator))
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli(*_broadcast(lhs, rhs.denominator))
            numerator = addi(*_broadcast(numerator, rhs.numerator))
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            lhs_numerator = muli(*_broadcast(lhs.numerator, rhs.denominator))
            rhs_numerator = muli(*_broadcast(rhs.numerator, lhs.denominator))
            numerator = addi(*_broadcast(lhs_numerator, rhs_numerator))
            denominator = muli(*_broadcast(lhs.denominator, rhs.denominator))
            return _Rational(numerator, denominator)
        else:
            return addi(*_broadcast(lhs, rhs))

    # `x * (a/b)` transformed into `(x * a) / b`
    def _mul(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli(*_broadcast(lhs.numerator, rhs))
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli(*_broadcast(lhs, rhs.numerator))
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            numerator = muli(*_broadcast(lhs.numerator, rhs.numerator))
            denominator = muli(*_broadcast(lhs.denominator, rhs.denominator))
            return _Rational(numerator, denominator)
        else:
            return muli(*_broadcast(lhs, rhs))

    def _floor(value):
        if isinstance(value, _Rational):
            value = arith_d.divsi(*_broadcast(value.numerator, value.denominator))

        return value

    def _ceiling(value):
        if isinstance(value, _Rational):
            if _emulate_ceildiv:
                # ceildivui(x, y) = x == 0 ? 0 : ((x - 1) / y) + 1
                one = _get_const(1)
                zero = _get_const(0)
                lhs_minus_one = arith_d.subi(*_broadcast(value.numerator, one))
                div = arith_d.divui(*_broadcast(lhs_minus_one, value.denominator))
                result = arith_d.addi(*_broadcast(div, one))
                cmp = arith_d.cmpi(
                    arith_d.CmpIPredicate.eq, *_broadcast(value.numerator, zero)
                )
                zero, result = _broadcast(zero, result)
                value = arith_d.select(cmp, zero, result)
            else:
                value = arith_d.ceildivsi(
                    *_broadcast(value.numerator, value.denominator)
                )

        return value

    def _group_rationals(stack, count):
        """Group rationals and non-rationals args into 2 contiguous sets.

        This allows to mul/add all non-rationals first, reducing total number of ops.
        """
        rationals = []
        non_rationals = []
        for _ in range(count):
            val = stack.pop()
            if isinstance(val, _Rational):
                rationals.append(val)
            else:
                non_rationals.append(val)

        return non_rationals + rationals

    def _apply(args, func):
        assert len(args) > 0
        value = args[0]
        for val in args[1:]:
            value = func(value, val)

        return value

    def _enforce_non_rational(val, term):
        if isinstance(val, _Rational):
            raise CodegenError(f"Rational is not supported yet in '{type(term)}'")

    def _remove_denominators(lhs, rhs):
        """
        Converts     (z/x) < y -> z < x*y
              or     z < (y/x) -> z*x < y
              or (a/b) < (c/d) -> a*d < b*c
        """
        if isinstance(lhs, _Rational) and not isinstance(rhs, _Rational):
            rhs = _mul(lhs.denominator, rhs)
            lhs = lhs.numerator
        if isinstance(rhs, _Rational) and not isinstance(lhs, _Rational):
            lhs = _mul(rhs.denominator, lhs)
            rhs = rhs.numerator
        if isinstance(lhs, _Rational) and isinstance(rhs, _Rational):
            rhs = _mul(lhs.denominator, rhs.numerator)
            lhs = _mul(rhs.denominator, lhs.numerator)
        return lhs, rhs

    def _get_const(val):
        if isinstance(val, int):
            return arith_d.constant(IndexType.get(), val)

        if isinstance(val, (tuple, list)):
            vec_type = VectorType.get([len(val)], IndexType.get())
            vals = [IntegerAttr.get(IndexType.get(), v) for v in val]
            return arith_d.constant(vec_type, DenseElementsAttr.get(vals, vec_type))

        raise CodegenError(f"Unsupported const val {val} : {type(val)}")

    idxc = IndexingContext.current()
    # Substitute in frozen vars to simplify expression.
    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)
    expr = expr.subs(idxc.subs)
    # Why affine, for now simply create indexing expressions.
    # This can easily be adapted to affine expressions later.
    select_stack = []
    if isinstance(expr, sympy.Piecewise):
        assert len(expr.args) == 2 and expr.args[1][1], f"Unsupported piecewise {expr}"
    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                res = idxc.get_val(term)
                if res is not None:
                    stack.append(_get_const(res))
                elif term in dynamics.keys():
                    stack.append(dynamics[term])
                else:
                    raise CodegenError(f"Unknown symbol {term}")
            case sympy.Integer():
                stack.append(_get_const(int(term)))
            case sympy.Mul():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _mul))
            case sympy.Add():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _add))
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                mod = arith_d.remsi(*_broadcast(lhs, rhs))
                stack.append(mod)
            case sympy.floor():
                stack.append(_floor(stack.pop()))
            case sympy.ceiling():
                stack.append(_ceiling(stack.pop()))
            case sympy.Rational():
                numerator = _get_const(term.p)
                denominator = _get_const(term.q)
                stack.append(_Rational(numerator, denominator))
            case sympy.LessThan():
                rhs = stack.pop()
                lhs = stack.pop()
                if isinstance(rhs, _Rational) or isinstance(lhs, _Rational):
                    lhs, rhs = _remove_denominators(lhs, rhs)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sle, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.StrictLessThan():
                rhs = stack.pop()
                lhs = stack.pop()
                if isinstance(rhs, _Rational) or isinstance(lhs, _Rational):
                    lhs, rhs = _remove_denominators(lhs, rhs)
                res = arith_d.cmpi(arith_d.CmpIPredicate.slt, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.StrictGreaterThan():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sgt, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.GreaterThan():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.sge, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.And():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.andi(*_broadcast(lhs, rhs))
                for _ in range(len(term.args) - 2):
                    operand = stack.pop()
                    _enforce_non_rational(operand, term)
                    res = arith_d.andi(*_broadcast(res, operand))
                stack.append(res)
            case sympy.Max():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                elem_type = get_type_or_element_type(rhs.type)
                if _is_integer_like_type(elem_type):
                    res = arith_d.maxsi(*_broadcast(lhs, rhs))
                else:
                    res = arith_d.maximumf(*_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.Min():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                elem_type = get_type_or_element_type(rhs.type)
                if _is_integer_like_type(elem_type):
                    res = arith_d.minsi(*_broadcast(lhs, rhs))
                else:
                    res = arith_d.minimumf(*_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.logic.boolalg.BooleanFalse():
                res = arith_d.constant(IntegerType.get_signless(1), 0)
                stack.append(res)
            case sympy.logic.boolalg.BooleanTrue():
                res = arith_d.constant(IntegerType.get_signless(1), 1)
                stack.append(res)
            case sympy.Pow():
                _, power = term.args
                exponent = stack.pop()
                base = stack.pop()
                # Only support integer powers for now.
                if not isinstance(power, sympy.Integer):
                    raise CodegenError(f"Expected integer power, got {power}")
                if power == 0:
                    stack.append(_get_const(1))
                    continue
                for _ in range(sympy.Abs(power) - 1):
                    base = arith_d.muli(base, base)
                if power < 0:
                    stack.append(_Rational(_get_const(1), base))
                else:
                    stack.append(base)
            case sympy.UnevaluatedExpr():
                continue
            case sympy.functions.elementary.piecewise.ExprCondPair():
                cond = stack.pop()
                expr = stack.pop()
                select_stack.append(cond)
                select_stack.append(expr)
                continue
            case sympy.Piecewise():
                expr = select_stack.pop()
                cond = select_stack.pop()
                last_expr = select_stack.pop()
                last_cond = select_stack.pop()
                res = arith_d.select(last_cond, *_broadcast(last_expr, expr))
                stack.append(res)
            case _:
                raise CodegenError(f"Can not handle {type(term)} : {term}")

    if len(stack) != 1 or isinstance(stack[0], _Rational):
        raise CodegenError(f"Expected single result, got {len(stack)}")

    return stack[0]


def get_constant_attr(value: Any, element_type: IrType) -> Attribute:
    if _is_integer_like_type(element_type):
        return IntegerAttr.get(element_type, int(value))
    if _is_float_type(element_type):
        return FloatAttr.get(element_type, float(value))
    raise CodegenError(f"Cannot create a constant attribute for type `{element_type}`")
