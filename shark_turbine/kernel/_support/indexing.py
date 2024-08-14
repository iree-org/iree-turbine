from typing import Any, ClassVar, Optional, Type, TypeVar, Union

from abc import ABC
from dataclasses import dataclass

import sympy

from . import context
from . import dtype
from .shaped_type import ShapedType, ShapedDataType

__all__ = [
    "backed_sym_index_type",
    "sym",
    "BoundedRelation",
    "EqualRelation",
    "IndexingContext",
    "IndexRelation",
    "IndexExpr",
    "IndexSequence",
    "IndexSymbol",
    "SymIndex",
]

DataType = dtype.DataType
DefaultDataType = dtype.f32


class NotSetType:
    ...


NotSet = NotSetType()

SubtypeT = TypeVar("SubtypeT")

###############################################################################
# Index symbols and expressions
# These are just light-weight helpers around sympy symbols and expressions.
###############################################################################

IndexSymbol = sympy.Symbol
IndexExpr = sympy.Expr


def index_symbol(name: str) -> IndexSymbol:
    """Returns a named symbol, assumed to be a non-negative integer."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n) -> IndexSymbol:
        return index_symbol(n)


sym = _IndexSymbolExpando()

###############################################################################
# Shape expressions
###############################################################################

SymbolicDimable = Union[str, IndexExpr]
SymbolicShapeable = tuple[SymbolicDimable]
SymbolicShapeExpr = tuple[IndexExpr]
Dims = list[Union[None, IndexSymbol, int]]

###############################################################################
# IndexingContext
###############################################################################


@dataclass(slots=True)
class _ShapedBinding:
    # The instance of shaped_type. Can be anything. We resolve dimension values
    # against this.
    instance: Any

    # Shaped type that backes the instance.
    shaped_type: ShapedType

    # The symbolic shape (tuple of index expressions).
    symbolic_shape: list[IndexExpr]

    # Concrete dimensions instantiated with. Each is an integer or a dynamic
    # dim symbol. It can also be None if the value is not dynamic and must be
    # inferred from context.
    dims: Dims


class IndexingContext:
    """The indexing context is responsible handling the binding of indexed
    symbols to concrete values.
    """

    __slots__ = [
        "subs",
        "shaped_bindings",
        "dyn_dims",
        "frozen_subs",
        "unbacked_symbols",
    ]

    __tk_context_idname__ = "IndexingContext"

    def __init__(self):
        self.subs: dict[IndexSymbol, int] = {}
        # Indexed by .instance
        self.shaped_bindings: dict[Any, _ShapedBinding] = {}
        self.dyn_dims: list[IndexSymbol] = []
        self.frozen_subs: list[tuple[IndexSymbol, int]] = []
        self.unbacked_symbols: list[IndexSymbol] = []

    def next_dyn_dim(self) -> IndexSymbol:
        s = index_symbol(f"D{len(self.dyn_dims)}")
        self.dyn_dims.append(s)
        return s

    def new_unbacked_symbol(self) -> IndexSymbol:
        s = index_symbol(f"_S{len(self.unbacked_symbols)}")
        self.unbacked_symbols.append(s)
        return s

    def bind_shaped(self, instance: Any, shaped_type: ShapedType, dims: Dims) -> None:
        if instance in self.shaped_bindings:
            raise ValueError(f"Argument binding {instance} is already bound")
        symbolic_shape = shaped_type.symbolic_shape
        rank = shaped_type.rank
        if rank != len(dims):
            raise ValueError(
                f"For {shaped_type} mismatched symbolic shape vs dim arity: {symbolic_shape} vs {dims}"
            )
        binding = _ShapedBinding(
            instance, shaped_type, list(symbolic_shape), list(dims)
        )
        self.shaped_bindings[instance] = binding

    def bind_constant(self, sym: IndexSymbol, value: int) -> None:
        try:
            self._bind_symbol(sym, value)
        except ValueError:
            raise ValueError(
                f"Attempt to bind symbol {sym}={value} conflicts with previous "
                f"{self.subs[sym]}"
            )

    def _bind_symbol(self, symbol: IndexSymbol, value: int):
        existing = self.subs.get(symbol)
        if existing is not None and existing != value:
            raise ValueError
        self.subs[symbol] = value

    def finalize(self):
        assert len(self.frozen_subs) == 0
        # Go over everything we know and bind all free symbols.
        for _sb in self.shaped_bindings.values():
            for i in range(_sb.shaped_type.rank):
                dim_expr = _sb.symbolic_shape[i]
                dim_value = _sb.dims[i]
                if dim_value is not None:
                    if isinstance(dim_expr, IndexSymbol):
                        try:
                            self._bind_symbol(dim_expr, dim_value)
                        except ValueError as e:
                            raise ValueError(
                                f"For {_sb.instance} of {_sb.shaped_type} attempt to bind dim "
                                f"{dim_expr}={dim_value} conflicts with previous "
                                f"{self.subs[dim_expr]}"
                            )

        # Note: At this point, we could solve the set of equation based
        # bindings and maybe elicit some additional information, but for now
        # we do forward-only inference.
        frozen_subs = self.frozen_subs
        frozen_subs.extend(self.subs.items())

        # Check any equation based dims.
        errors = []
        for _sb in self.shaped_bindings.values():
            for i in range(_sb.shaped_type.rank):
                dim_expr = _sb.symbolic_shape[i]
                dim_value = _sb.dims[i]
                dim_expr = dim_expr.subs(frozen_subs).simplify()
                _sb.symbolic_shape[i] = dim_expr
                if dim_value is None:
                    # Ensure resolves to a known value.
                    if not isinstance(dim_expr, sympy.Integer):
                        errors.append(
                            f"  {_sb.instance} of {_sb.shaped_type}[{i}]={dim_expr} did not "
                            f"resolve to a known value"
                        )
                        continue
                    # Notate the inferred dim.
                    _sb.dims[i] = int(dim_expr)
                elif isinstance(dim_expr, sympy.Integer):
                    dim_expr_value = int(dim_expr)
                    if isinstance(dim_value, IndexExpr):
                        # If dynamic, then it turns out we have enough static information,
                        # so replace.
                        _sb.dims[i] = dim_expr_value
                    else:
                        # If static, make sure it matches the runtime value.
                        if dim_value is not None and dim_expr_value != dim_value:
                            errors.append(
                                f"  {_sb.instance} of {_sb.shaped_type}[{i}]={dim_expr} was initialized with a "
                                f"mismatched runtime value of {dim_value}"
                            )
                            continue

        # Error check.
        if errors:
            joined = "\n".join(errors)
            raise ValueError(f"Indexing mismatches were encountered:\n{joined}")

    def eval_dim(self, instance: Any, shaped_type: ShapedType, pos: int) -> IndexExpr:
        # TODO: Could see if shaped_type is in self.shaped_bindings: it has some
        # precomputed values that may save cycles to use.
        symbolic_shape = shaped_type.symbolic_shape
        try:
            expr = symbolic_shape[pos]
        except IndexError:
            raise IndexError(f"Attempt to access out of range {shaped_type}[{pos}]")
        return expr.subs(self.frozen_subs).simplify()

    def eval_static_dim(
        self, instance: Any, shaped_type: ShapedType, pos: int
    ) -> Optional[int]:
        expr = self.eval_dim(instance, shaped_type, pos)
        try:
            return int(expr)
        except TypeError:
            return None

    def simplify_expr(self, expr: IndexExpr) -> IndexExpr:
        return expr.subs(self.frozen_subs).simplify()

    def get_static_value(self, expr: IndexExpr) -> Optional[int]:
        expr = self.simplify_expr(expr)
        try:
            return int(expr)
        except TypeError:
            return None

    ##### Context management.
    @staticmethod
    def current() -> "IndexingContext":
        return context.current(IndexingContext)

    def __enter__(self) -> "IndexingContext":
        return context.push(IndexingContext, self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        context.pop(IndexingContext, self)


###############################################################################
# Symbolic index value type.
# TODO: We think we want to remove this in the next rev, in favor of doing
# relationship verification as part of a pass.
###############################################################################


class IndexRelation(ABC):
    """ABC for assumptions that can be made about an index value."""

    __slots__ = []


class EqualRelation(IndexRelation):
    """An index assumption that can take a single symbolic value."""

    __slots__ = ["eq_expr"]

    def __init__(self, eq_expr: IndexExpr):
        self.eq_expr = eq_expr

    def __eq__(self, other):
        if not isinstance(other, EqualRelation):
            return False
        return self.eq_expr == other.eq_expr

    def __repr__(self):
        expr = self.eq_expr
        if isinstance(expr, IndexSymbol):
            return f"=={expr}"
        else:
            return f"==({expr})"


class BoundedRelation(IndexRelation):
    """An index assumption that can take any value in a range."""

    __slots__ = [
        "lower_expr",
        "lower_inclusive",
        "upper_expr",
        "upper_inclusive",
    ]

    def __init__(
        self,
        lower_expr: Any,
        upper_expr: Any,
        *,
        lower_inclusive: bool = True,
        upper_inclusive: bool = True,
    ):
        self.lower_expr = index_expr(lower_expr)
        self.lower_inclusive = lower_inclusive
        self.upper_expr = index_expr(upper_expr)
        self.upper_inclusive = upper_inclusive

    def __eq__(self, other):
        if not isinstance(other, BoundedRelation):
            return False
        return (
            self.lower_inclusive == other.lower_inclusive
            and self.upper_inclusive == other.upper_inclusive
            and self.lower_expr == other.lower_expr
            and self.upper_expr == other.upper_expr
        )

    def __repr__(self):
        return (
            f"∈{'[' if self.lower_inclusive else '('}"
            f"{self.lower_expr}, {self.upper_expr}"
            f"{']' if self.upper_inclusive else ')'}"
        )


class _SymIndexMeta(type):
    """Meta-class for a concrete symbolic index value."""

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
        *,
        assumption: Optional[IndexRelation],
    ):
        new_class = type.__new__(mcls, name, bases, dct)
        new_class.assumption = assumption
        new_class.__qualname__ = repr(new_class)
        return new_class

    def __repr__(self):
        if self.assumption:
            return f"SymIndex{self.assumption}"
        else:
            return "UnbackedSymIndex"


class SymIndex(metaclass=_SymIndexMeta, assumption=None):
    """Symbolic index value defined for an assumption.

    The base type is unbacked (None assumption).
    """

    __slots__ = [
        "symbol",
    ]

    assumption: ClassVar[Optional[IndexRelation]]

    def __init__(self, symbol: IndexSymbol):
        self.symbol = symbol

    def __repr__(self):
        return f"<'{self.symbol}' over {type(self)}>"

    def cast(self, cast: Type["SymIndex"]) -> "SymIndex":
        """Cast the SymIndex to a new type, typically to further constrain it.

        The new instance shares the symbol.
        """
        return cast(self.symbol)


def backed_sym_index_type(assumption: IndexRelation) -> Type[SymIndex]:
    class BackedSymIndex(SymIndex, assumption=assumption):
        ...

    return BackedSymIndex


@dataclass
class IndexSequence:
    start: IndexExpr | int
    size: IndexExpr | int
    stride: Optional[IndexExpr | int] = 1

    def _subs(
        self, value: int | IndexExpr, map: dict[IndexSymbol, int]
    ) -> int | IndexExpr:
        new_value = value
        if isinstance(value, IndexExpr):
            new_value = value.subs(map)
        return new_value

    def subs(self, map: dict[IndexSymbol, int]):
        start = self._subs(self.start, map)
        size = self._subs(self.size, map)
        stride = self._subs(self.stride, map)
        return IndexSequence(start, size, stride)

    def __repr__(self):
        if isinstance(self.size, sympy.Integer):
            self.size = int(self.size)
        if isinstance(self.size, int) and self.size <= 1:
            return f"{self.start}"
        return f"{self.start} : {self.size} : {self.stride}"
