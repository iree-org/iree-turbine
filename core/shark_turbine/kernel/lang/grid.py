from typing import cast, Type, ClassVar

from .._support.shaped_type import ShapedType
from .._support.indexing import IndexingContext, IndexExpr

__all__ = [
    "Grid",
]


class Grid(metaclass=ShapedType):
    """Grid with bounding symbolic shape information in the type."""

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dims: list[int]

    def __init__(self):
        # Resolve the symbolic shape to concrete values.
        idxc = IndexingContext.current()
        if self.symbolic_shape:
            dims = [idxc.get_static_value(dim) for dim in self.symbolic_shape]
            if None in dims:
                raise ValueError(f"NYI: Dynamic dims in Grid")
            self.dims = cast(list[int], dims)
        else:
            self.dims = []

    def __class_getitem__(
        cls, symbolic_shape: tuple[IndexExpr, ...] | IndexExpr
    ) -> Type["Grid"]:
        if not isinstance(symbolic_shape, tuple):
            symbolic_shape = (symbolic_shape,)

        return cls.new_shaped_subtype(symbolic_shape=symbolic_shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dims)

    def __repr__(self):
        return f"{repr(type(self))}({', '.join(str(i) for i in self.dims)})"

    def __getitem__(self, index: int) -> int:
        return self.dims[index]

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)
