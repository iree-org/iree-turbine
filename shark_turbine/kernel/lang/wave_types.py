from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Optional,
    Self,
    Type,
    TypeAlias,
    TypeVar,
)

from .kernel_buffer import AddressSpace, KernelBufferMeta, KernelBufferUsage
from ..ops.wave_ops import register
from .._support.dtype import DataType
from .._support.indexing import IndexExpr, IndexSymbol, index_symbol

from sympy import Symbol
from sympy.core.expr import Expr

from itertools import chain

__all__ = [
    "IndexMapping",
    "Memory",
    "Register",
]

MemoryTypeT = TypeVar("MemoryTypeT")


class Memory(metaclass=KernelBufferMeta):
    """
    Represents storage anywhere in the memory hierarchy except registers.
    Parameterized by a shape, address space and element type. The allocated
    memory is traversed by an iterator that specifies the offset, stride
    and size along each dimension.
    """

    address_space: ClassVar[int]
    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    usage: ClassVar[Optional[KernelBufferUsage]]

    def __init__(self) -> None:
        raise NotImplementedError("Memory types are not directly instantiated.")

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Memory"]:
        """Syntax: `Memory[shape1, ...., shapeN, addressSpace, dtype, Optional[usage]]"""
        if len(shape_and_dtype) < 3:
            raise TypeError(f"Expected at least 3 arguments, got: {shape_and_dtype}")

        shift = 0
        usage = KernelBufferUsage.NONE
        if isinstance(shape_and_dtype[-1], KernelBufferUsage):
            shift = 1
            usage = shape_and_dtype[-1]
        shape = shape_and_dtype[: -2 - shift]
        addressSpace = shape_and_dtype[-2 - shift]
        dtype = shape_and_dtype[-1 - shift]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)
        if not all(isinstance(s, IndexExpr) for s in shape) or len(shape) == 0:
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")
        if not (
            isinstance(addressSpace, IndexExpr)
            or isinstance(addressSpace, AddressSpace)
        ):
            raise TypeError(
                f"Expected addressSpace to be a AddressSpace, got {addressSpace}"
            )
        if addressSpace == AddressSpace.REGISTER:
            raise TypeError(
                f"Memory does not support address space register, use Register instead."
            )

        return cls.new_subtype(
            name="Memory",
            address_space=addressSpace,
            symbolic_shape=shape,
            dtype=dtype,
            usage=usage,
        )


class Register(metaclass=KernelBufferMeta):
    """
    Represents virtual registers. Parameterized by a shape and element type.
    Instantiating this class emits a new `register` operation.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    value: float

    def __new__(cls, value: float) -> "Register":
        return register(cls.symbolic_shape, cls.dtype, value)

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Register"]:

        if len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)

        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")

        return cls.new_subtype(
            name="Register",
            address_space=AddressSpace.REGISTER,
            symbolic_shape=shape,
            dtype=dtype,
        )


SymbolsMap: TypeAlias = dict[IndexSymbol, IndexExpr]


def _subs_expr(expr: Any, subs: Iterable[tuple[IndexExpr, IndexExpr]]) -> Any:
    if isinstance(expr, (Symbol, Expr)):
        return expr.subs(subs)

    return expr


def _is_identity_mapping(iters: Iterable[IndexSymbol], mapping: SymbolsMap) -> bool:
    if len(iters) != len(mapping):
        return False

    for it, val in zip(iters, mapping.values()):
        if it != val:
            return False

    return True


class IndexMapping:
    """
    Represents a mapping between 2 sets of indices.
    """

    iters: dict[IndexSymbol, int]
    input_mapping: SymbolsMap
    output_mapping: SymbolsMap
    iteration_shape: tuple[IndexExpr, ...]

    def __init__(
        self, num_iterators: int, inputs: SymbolsMap, outputs: SymbolsMap
    ) -> None:
        iters = {self.iterator(i): i for i in range(num_iterators)}
        iter_shape = [None] * num_iterators
        for sym, expr in chain(inputs.items(), outputs.items()):
            i = iters.get(expr, None)
            if i is None:
                continue

            current = iter_shape[i]
            assert (
                current is None or current == sym
            ), f"Iterator conflict: {current} and {sym}"
            iter_shape[i] = sym

        assert all(
            i is not None for i in iter_shape
        ), "Cannot determine iteration domain"
        self.iters = iters
        self.iteration_shape = iter_shape
        self.input_mapping = inputs
        self.output_mapping = outputs

    @property
    def num_iterators(self) -> int:
        return len(self.iters)

    def substitute(self, subs: Iterable[tuple[IndexExpr, IndexExpr]]) -> Self:
        new_inputs = {
            key: _subs_expr(val, subs) for key, val in self.input_mapping.items()
        }
        new_outputs = {
            key: _subs_expr(val, subs) for key, val in self.output_mapping.items()
        }
        return IndexMapping(self.num_iterators, new_inputs, new_outputs)

    @property
    def output_shape(self) -> tuple[IndexExpr]:
        return tuple(self.output_mapping.keys())

    @staticmethod
    def iterator(index: int) -> IndexSymbol:
        return index_symbol(f"$index{index}")

    def _map_indices(
        self, mapping: SymbolsMap, symbols: Optional[tuple[IndexSymbol, ...]]
    ) -> tuple[IndexExpr, ...]:
        if symbols is None:
            return tuple(mapping.values())

        return tuple(mapping[sym] for sym in symbols)

    def map_input_indices(
        self, symbols: Optional[tuple[IndexSymbol, ...]] = None
    ) -> tuple[IndexExpr, ...]:
        return self._map_indices(self.input_mapping, symbols)

    def map_output_indices(
        self, symbols: Optional[tuple[IndexSymbol, ...]] = None
    ) -> tuple[IndexExpr, ...]:
        return self._map_indices(self.output_mapping, symbols)

    def is_output_identity(self) -> bool:
        return _is_identity_mapping(self.iters.keys(), self.output_mapping)
