from dataclasses import dataclass
from typing import Optional
from .global_symbols import THREAD_0, THREAD_1, THREAD_2
from .._support.indexing import IndexingContext, IndexExpr, safe_subs
from sympy import Integer


@dataclass
class ThreadBlock:
    """Thread block with bounding symbolic shape information."""

    symbolic_shape: tuple[IndexExpr, IndexExpr, IndexExpr]
    static_shape: Optional[tuple[int, int, int]] = None

    def infer_static_shape(self, threads_per_wave: int, idxc: IndexingContext):
        """Infer the static shape of the thread block using the indexing context."""
        self.static_shape = tuple(
            idxc.get_static_value(safe_subs(sym, idxc.subs))
            for sym in self.symbolic_shape
        )
        if None in self.static_shape:
            raise ValueError(f"NYI: Dynamic dims in block")
        if self.static_shape[0] % threads_per_wave != 0:
            raise ValueError(
                "The first dimension is not divisible by the number of threads in a wave"
            )

    @property
    def linearized_thread_id(self) -> IndexExpr:
        thread_ids = [THREAD_0, THREAD_1, THREAD_2]
        symbolic_shape = [
            1,
            self.static_shape[0],
            self.static_shape[0] * self.static_shape[1],
        ]
        return sum([x * y for x, y in zip(thread_ids, symbolic_shape)])


def dim3_from_num_waves(
    threads_per_wave: int,
    x: int | IndexExpr,
    y: int | IndexExpr = 1,
    z: int | IndexExpr = 1,
) -> ThreadBlock:
    """Creates a thread block from x, y, z num waves"""
    get = lambda v: Integer(v) if isinstance(v, int) else v
    return ThreadBlock((threads_per_wave * get(x), get(y), get(z)))
