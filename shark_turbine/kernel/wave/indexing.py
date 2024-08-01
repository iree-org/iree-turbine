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
