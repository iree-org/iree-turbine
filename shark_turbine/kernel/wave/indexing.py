from dataclasses import dataclass
from typing import Optional, Any
from ...support.logging import get_logger
from .._support.indexing import IndexExpr, IndexSymbol
import sympy

logger = get_logger("turbine.wave.indexing")


@dataclass
class IndexSequence:
    start: IndexExpr | int
    size: IndexExpr | int
    stride: Optional[IndexExpr | int] = 1

    def __add__(self, other: Any) -> Any:
        if isinstance(other, IndexSequence):
            return IndexSequence(
                self.start + other.start,
                self.size * other.size,
                self.stride * other.stride,
            )
        else:
            raise NotImplementedError("IndexSequence addition not implemented!")

    def subs(self, map: dict[IndexSymbol, int]):
        start = self.start
        if isinstance(self.start, IndexExpr):
            start = self.start.subs(map)
        size = self.size
        if isinstance(self.size, IndexExpr):
            size = self.size.subs(map)
        stride = self.stride
        if isinstance(self.stride, IndexExpr):
            stride = self.stride.subs(map)
        return IndexSequence(start, size, stride)

    def __repr__(self):
        if isinstance(self.size, sympy.Integer):
            self.size = int(self.size)
        if isinstance(self.size, int) and self.size <= 1:
            return f"{self.start}"
        return f"{self.start} : {self.size} : {self.stride}"
