import typing
from typing import Optional, Type, TypeVar, cast

from .dtype import DataType

if typing.TYPE_CHECKING:
    from .indexing import IndexExpr

SymbolicShapeExpr = tuple["IndexExpr", ...]

SubtypeT = TypeVar("SubtypeT")

###############################################################################
# Shaped Type
###############################################################################


def _shaped_data_type_repr(
    name: str,
    *,
    symbolic_shape: Optional[SymbolicShapeExpr],
    dtype: Optional[DataType] = None,
) -> str:
    stem = name
    if symbolic_shape:
        stem += f"[{', '.join(repr(s) for s in symbolic_shape)}]"
    if dtype:
        stem += f".of({dtype})"
    return stem


class ShapedType(type):
    """A shaped type.

    This lets us specialize with symbolic shape information.
    """

    symbolic_shape: Optional[SymbolicShapeExpr] = None
    rank: Optional[int]

    def __new__(mcls, name: str, bases, dct):
        symbolic_shape = dct.get("symbolic_shape")
        if symbolic_shape is not None:
            rank = len(symbolic_shape)
            dct["rank"] = rank

        # TODO: I don't know a better way to do this. Ask Stella for better way.
        if "__qualname__" not in dct:
            dct["__qualname__"] = _shaped_data_type_repr(
                name, symbolic_shape=symbolic_shape
            )

        new_class = type.__new__(mcls, name, bases, dct)
        return new_class

    def new_shaped_subtype(
        cls: Type[SubtypeT],
        *,
        symbolic_shape: SymbolicShapeExpr,
    ) -> Type[SubtypeT]:
        init_symbolic_shape = symbolic_shape

        class Subtype(cls):
            symbolic_shape = init_symbolic_shape
            rank = len(init_symbolic_shape)

        Subtype.__name__ = cls.__name__

        return cast(Type[SubtypeT], Subtype)

    def __str__(cls):
        return repr(cls)

    def __repr__(cls):
        return _shaped_data_type_repr(cls.__name__, symbolic_shape=cls.symbolic_shape)


###############################################################################
# Shaped Data Type
###############################################################################


class ShapedDataType(ShapedType):
    """A shaped type containing data of a specific element type.

    This lets us specialize with symbolic shape information.
    """

    dtype: Optional[DataType] = None

    def __new__(
        mcls,
        name: str,
        bases,
        dct,
    ):
        shaped_type = dct.get("shaped_type")
        dtype = dct.get("dtype")

        if "__qualname__" not in dct:
            dct["__qualname__"] = _shaped_data_type_repr(
                name,
                symbolic_shape=shaped_type,
                dtype=dtype,
            )

        new_class = type.__new__(mcls, name, bases, dct)
        return new_class

    def new_shaped_data_subtype(
        cls: Type[SubtypeT],
        *,
        symbolic_shape: SymbolicShapeExpr,
        dtype: DataType,
    ) -> Type[SubtypeT]:
        init_symbolic_shape = symbolic_shape
        init_dtype = dtype

        class Subtype(cls):
            symbolic_shape = init_symbolic_shape
            rank = len(init_symbolic_shape)
            dtype = init_dtype

        Subtype.__name__ = cls.__name__

        return cast(Type[SubtypeT], Subtype)

    def __str__(cls):
        return repr(cls)

    def __repr__(cls):
        return _shaped_data_type_repr(
            cls.__name__,
            symbolic_shape=cls.symbolic_shape,
            dtype=cls.dtype,
        )
