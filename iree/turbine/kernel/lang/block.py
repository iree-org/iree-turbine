from dataclasses import dataclass
from math import prod

from .global_symbols import THREAD_0, THREAD_1, THREAD_2
from .._support.indexing import IndexExpr


@dataclass
class ThreadBlock:
    """Thread block with bounding symbolic shape information."""

    shape: tuple[int, int, int]

    def __post_init__(self):
        assert 0 < prod(self.shape) <= 1024, "Invalid thread block."

    @property
    def linearized_thread_id(self) -> IndexExpr:
        thread_ids = [THREAD_0, THREAD_1, THREAD_2]
        shape = [
            1,
            self.shape[0],
            self.shape[0] * self.shape[1],
        ]
        return sum([x * y for x, y in zip(thread_ids, shape)])

    def linearized_wave_id(self, threads_per_wave: int) -> IndexExpr:
        """Returns a linearized wave id."""
        ids = [THREAD_0 // threads_per_wave, THREAD_1, THREAD_2]
        x_sz, y_sz = (self.shape[0] // threads_per_wave, self.shape[1])
        shape = (1, x_sz, x_sz * y_sz)
        non_zero_ids = [i for i, dim in enumerate(self.shape) if dim > 1]
        return sum([ids[i] * shape[i] for i in non_zero_ids])


def dim3_from_num_waves(
    threads_per_wave: int,
    x: int,
    y: int = 1,
    z: int = 1,
) -> ThreadBlock:
    """Creates a thread block from x, y, z num waves"""
    return ThreadBlock((threads_per_wave * x, y, z))
