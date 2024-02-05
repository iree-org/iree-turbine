from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import sympy
import torch

from .. import ops

from . import context
from . import dtype

__all__ = [
    "backed_sym_index_type",
    "sym",
    "BoundedRelation",
    "EqualRelation",
    "Grid",
    "IndexingContext",
    "IndexRelation",
    "IndexExpr",
    "IndexSymbol",
    "InputBuffer",
    "KernelBuffer",
    "OutputBuffer",
    "SymIndex",
    "TemporaryBuffer",
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

IndexSymbol = sympy.core.Symbol
IndexExpr = sympy.core.Expr


def index_symbol(name: str) -> IndexSymbol:
    """Returns a named symbol, assumed to be a non-negative integer."""
    return sympy.Symbol(name, integer=True, nonnegative=True)


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n):
        return index_symbol(n)


sym = _IndexSymbolExpando()

###############################################################################
# Shape expressions
###############################################################################

SymbolicDimable = Union[str, IndexExpr]
SymbolicShapeable = tuple[SymbolicDimable]
SymbolicShapeExpr = tuple[IndexExpr]


def make_symbolic_shape(elements: SymbolicShapeable) -> SymbolicShapeExpr:
    return tuple(
        index_symbol(expr) if isinstance(expr, str) else expr for expr in elements
    )


###############################################################################
# Grid
###############################################################################


class _GridMeta(type):
    """Meta-class for a symbolically shaped grid."""

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
        *,
        symbolic_shape: Optional[SymbolicShapeExpr],
    ):
        new_class = type.__new__(mcls, name, bases, dct)
        new_class.symbolic_shape = symbolic_shape
        new_class.rank = len(symbolic_shape) if symbolic_shape is not None else None
        new_class.__qualname__ = repr(new_class)
        return new_class

    def __class_getitem__(
        cls, symbolic_shape: Union[SymbolicDimable, tuple[SymbolicShapeable]]
    ) -> Type["Grid"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)
        return cast(Grid, _make_shaped_grid(cls, make_symbolic_shape(symbolic_shape)))

    def __repr__(self):
        if self.symbolic_shape:
            return f"Grid[{', '.join(repr(s) for s in self.symbolic_shape)}]"
        else:
            return "Grid"


class Grid(metaclass=_GridMeta, symbolic_shape=None):
    """Grid with bounding symbolic shape information in the type."""

    symbolic_shape: ClassVar[Optional[SymbolicShapeExpr]]
    rank: int

    def __init__(self, *dims: int):
        rank = len(dims)
        if self.symbolic_shape is not None:
            if rank != len(self.symbolic_shape):
                raise ValueError(
                    f"Cannot create {type(self)}({', '.join(str(i) for i in dims)}): mismatched symbolic rank"
                )

        self.dims = dims
        # Shadow the type rank with the actual, which makes it concrete
        # for the generic case.
        self.rank = rank

    def __repr__(self):
        return f"{repr(type(self))}({', '.join(str(i) for i in self.dims)})"

    def __getitem__(self, index: int) -> int:
        return self.dims[index]

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)


def _make_shaped_grid(cls: Type[Grid], symbolic_shape: tuple[IndexExpr]):
    class ShapedGrid(Grid, symbolic_shape=symbolic_shape):
        ...

    return ShapedGrid


###############################################################################
# KernelBuffer
###############################################################################


class KernelBufferUsage(Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2
    TEMPORARY = 3

    @staticmethod
    def _type_name(v) -> str:
        if v == KernelBufferUsage.NONE:
            return "KernelBuffer"
        elif v == KernelBufferUsage.INPUT:
            return "InputBuffer"
        elif v == KernelBufferUsage.OUTPUT:
            return "OutputBuffer"
        elif v == KernelBufferUsage.TEMPORARY:
            return "TemporaryBuffer"
        else:
            raise AssertionError(f"uncovered KernelBufferUsage enum ({v})")


class _KernelBufferMeta(type):
    """Meta-class for kernel buffers.

    This lets us specialize with symbolic shape information.
    """

    element_type: DataType
    usage: KernelBufferUsage
    symbolic_shape: Optional[SymbolicShapeExpr]
    rank: Optional[int]

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
    ):
        element_type = dct.get("element_type") or DefaultDataType
        dct["element_type"] = element_type
        usage = dct.get("usage") or KernelBufferUsage.NONE
        dct["usage"] = usage
        if "usage" not in dct:
            dct["usage"] = KernelBufferUsage.NONE
        symbolic_shape = dct.get("symbolic_shape")
        dct["symbolic_shape"] = symbolic_shape
        dct["rank"] = len(symbolic_shape) if symbolic_shape is not None else None
        dct["__qualname__"] = _kernel_buffer_type_repr(
            element_type=element_type, usage=usage, symbolic_shape=symbolic_shape
        )
        new_class = type.__new__(mcls, name, bases, dct)
        return new_class

    def new_subtype(
        cls: Type[SubtypeT],
        *,
        element_type: Union[NotSetType, DataType] = NotSet,
        symbolic_shape: Union[NotSetType, Optional[SymbolicShapeable]] = NotSet,
        usage: Union[NotSetType, KernelBufferUsage] = NotSet,
    ) -> Type[SubtypeT]:
        init_element_type = (
            element_type if element_type is not NotSet else cls.element_type
        )
        init_symbolic_shape = (
            symbolic_shape if symbolic_shape is not NotSet else cls.symbolic_shape
        )
        init_usage = usage if usage is not NotSet else cls.usage

        class Subtype(cls):
            element_type = init_element_type
            symbolic_shape = make_symbolic_shape(init_symbolic_shape)
            usage = init_usage

        return Subtype

    def of(cls: Type[SubtypeT], element_type: Union[Any, DataType]) -> Type[SubtypeT]:
        return cls.new_subtype(element_type=element_type)

    def __repr__(cls):
        return _kernel_buffer_type_repr(
            element_type=cls.element_type,
            usage=cls.usage,
            symbolic_shape=cls.symbolic_shape,
        )


def is_kernel_buffer_meta_derived(t: type) -> bool:
    return isinstance(t, _KernelBufferMeta)


def _kernel_buffer_type_repr(
    *,
    element_type: DataType,
    usage: KernelBufferUsage,
    symbolic_shape: Optional[tuple[IndexExpr]],
) -> str:
    root = KernelBufferUsage._type_name(usage)
    if symbolic_shape:
        stem = f"{root}[{', '.join(repr(s) for s in symbolic_shape)}]"
    else:
        stem = f"{root}"
    if element_type != DefaultDataType:
        stem += f".of({element_type})"
    return stem


class KernelBuffer(metaclass=_KernelBufferMeta):
    """Represents a buffer in global memory.

    Top level kernels always operate on global memory via these
    buffers, and the primary operations that can be performed on
    them are loads/stores and DMAs to some form of compute
    capable local buffer.

    When executing eagerly, these are backed by a normal torch
    Tensor. When compiling, an appropriate duck-typed proxy
    is used.
    """

    usage: ClassVar[KernelBufferUsage]
    symbolic_shape: ClassVar[Optional[SymbolicShapeExpr]]
    rank: Optional[int]

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        type_rank = type(self).rank
        tensor_rank = len(tensor.shape)
        if type_rank is not None and type_rank != tensor_rank:
            raise ValueError(
                f"Cannot create {type(self)}(tensor({tensor.shape})): mismatched symbolic rank"
            )
        self._tensor = tensor
        self.rank = tensor_rank

    def __class_getitem__(
        cls, symbolic_shape: Union[IndexExpr, SymbolicShapeExpr]
    ) -> Type["KernelBuffer"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)
        return cast(
            cls, cls.new_subtype(symbolic_shape=make_symbolic_shape(symbolic_shape))
        )

    def __repr__(self):
        return f"{type(self)}({self._tensor})"

    def __setitem__(self, key, item):
        ops.kernel_buffer_setitem(self, key, item)

    def __getitem__(self, key):
        return ops.kernel_buffer_getitem(self, key)


class InputBuffer(KernelBuffer):
    usage = KernelBufferUsage.INPUT


class OutputBuffer(KernelBuffer):
    usage = KernelBufferUsage.OUTPUT


class TemporaryBuffer(KernelBuffer):
    usage = KernelBufferUsage.TEMPORARY


###############################################################################
# IndexingContext
###############################################################################

ShapedType = Union[Type[KernelBuffer], Type[Grid]]
Dims = list[Union[None, IndexSymbol, int]]


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
        self.frozen_subs: list[IndexSymbol, int] = []
        self.unbacked_symbols: list[IndexSymbol] = []

    def next_dyn_dim(self) -> IndexSymbol:
        s = index_symbol(f"D{len(self.dyn_dims)}")
        self.dyn_dims.append(s)
        return s

    def new_unbacked_symbol(self) -> IndexSymbol:
        s = index_symbol(f"_S{len(self.unbacked_symbols)}")
        self.unbacked_symbols.append(s)
        return s

    def bind_shaped(
        self, instance: Any, shaped_type: ShapedType, dims: Dims
    ) -> _ShapedBinding:
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

    def bind_constant(self, sym: IndexSymbol, value: int):
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
