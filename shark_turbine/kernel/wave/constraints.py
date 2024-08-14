from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from sympy import ceiling, Piecewise, floor

from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from ..lang.global_symbols import *


class MMAType(Enum):
    F32_16x16x16_F16 = 0
    F32_32x32x8_F16 = 1


@dataclass
class Constraint(ABC):
    """
    Base class for constraints. Every constraint reduces to
    the following form:
        Variables: [x0, x1, ...., xN]
        Bounds: [lb0 <= x0 <= ub0, ..., lbN <= xN <= ubN]
        Equality Constraints: [f0(x0, ..., xN) = 0, f1(x0, ..., xN) = 0, ...]
        Inequality Constraints: [g0(x0, ..., xN) <= 0, g1(x0, ..., xN) <= 0, ...]
    """

    @abstractmethod
    def apply(self) -> IndexSequence:
        """Apply the constraint and get the resulting index sequence."""
        ...


@dataclass
class HardwareConstraint(Constraint):
    """
    A constraint of the form
        tkw.HardwareConstraint(threads_per_wave = N,
                               mma_type = 'MFMA_F32_16x16x16_F16')
    specifies that the hardware supports N threads per wave and that
    we want all mma operations in the microkernel to be
    mapped to a hardware mma instruction of shape (16x16x16).
    This translates to a hardware specific index constraint.

    Not all computation graphs have mma operators in them. In
    these situations, the user can specify the vector shape they
    want to tile to by specifying the vector shapes dictionary
    which maps a tensor dimension to its corresponding tile size.
    """

    threads_per_wave: int
    waves_per_block: Optional[tuple[int, int, int]] = None
    mma_type: Optional[MMAType] = MMAType.F32_16x16x16_F16
    vector_shapes: Optional[dict[IndexSymbol, int]] = None

    @property
    def mma_matrix_shapes(self) -> tuple[int]:
        # TODO: Eventually the shapes and indices should be provided by a tool
        match self.mma_type:
            case MMAType.F32_16x16x16_F16:
                return (16, 16, 16)
            case MMAType.F32_32x32x8_F16:
                return (32, 32, 8)
            case _:
                return ()

    @property
    def threads_per_block(self) -> tuple[int]:
        return (
            self.waves_per_block[0] * self.threads_per_wave,
        ) + self.waves_per_block[1:]

    @property
    def linearized_thread_id(self) -> IndexExpr:
        thread_ids = [THREAD_0, THREAD_1, THREAD_2]
        threads_per_block = [
            1,
            self.threads_per_block[0],
            self.threads_per_block[0] * self.threads_per_block[1],
        ]
        return sum([x * y for x, y in zip(thread_ids, threads_per_block)])

    def apply(self, mma_index: int) -> IndexSequence:
        lane = self.linearized_thread_id
        match self.mma_type:
            # (M x K, N x K) -> M x N
            case MMAType.F32_16x16x16_F16:
                offset = [
                    Piecewise(
                        (lane % 16, ~MMA_ACC), (4 * floor(lane / 16), MMA_ACC)
                    ),  # M
                    lane % 16,  # N
                    4 * floor(lane / 16),  # K
                ]
                size = [
                    Piecewise((1, ~MMA_ACC), (4, MMA_ACC)),  # M
                    1,  # N
                    4,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
                return IndexSequence(
                    offset[mma_index], size[mma_index], stride[mma_index]
                )


@dataclass
class WorkgroupConstraint(Constraint):
    """
    A constraint of the form `tkw.WorkgroupConstraint(M, BLOCK_M, 0)`
    specifies that we want to distribute dimension M along workgroup dim 0
    with a tile size of BLOCK_M resulting in M // BLOCK_M workgroups along that
    dimension. This translates to an index constraint for all tensors of the
    shape [M, ?] -> index += (workgroup_id_0 * BLOCK_M, 0)
    """

    dim: IndexExpr
    tile_size: IndexExpr
    workgroup_dim: int

    def apply(self) -> IndexSequence:
        match self.workgroup_dim:
            case 0:
                wg_dim = WORKGROUP_0
            case 1:
                wg_dim = WORKGROUP_1
            case 2:
                wg_dim = WORKGROUP_2
            case _:
                raise ValueError("Invalid workgroup dimension. Expected 0, 1 or 2.")
        return IndexSequence(wg_dim * self.tile_size, 1)


def get_grid_shape(wg_constraints: list[WorkgroupConstraint]) -> list[IndexExpr]:
    sorted_constraints = sorted(wg_constraints, key=lambda x: x.workgroup_dim)
    # Currently not more than one constraint in each dimension supported.
    if any(
        sorted_constraints[i].workgroup_dim == sorted_constraints[i + 1].workgroup_dim
        for i in range(len(sorted_constraints) - 1)
    ):
        raise ValueError(
            "Multiple constraints in the same workgroup dimension are currently not supported."
        )
    grid: list[IndexExpr] = [
        constraint.dim // constraint.tile_size for constraint in wg_constraints
    ]
    return grid


@dataclass
class TilingConstraint(Constraint):
    """
    A constraint of the form `tkw.TilingConstraint(K, BLOCK_K)` specifies
    that we want to tile the K dimension with a tile size of BLOCK_K. This
    adds an index constraint to the K-th dimension of a tensor of the form
    BLOCK_K * i, where i is the induction variable associated with the
    loop around dimension K.
    """

    dim: IndexExpr
    tile_size: IndexExpr
    induction_var: Optional[IndexExpr] = None

    def iterations(self) -> IndexExpr:
        """
        Returns an expression for the number of iterations in the loop.
        """
        return ceiling(self.dim / self.tile_size)

    def apply(self) -> IndexSequence:
        if self.induction_var is None:
            raise ValueError(
                "Index is being computed without setting induction variable"
            )
        return IndexSequence(self.induction_var * self.tile_size, 1)


@dataclass
class WaveConstraint(Constraint):
    """
    A constraint of the form `tkw.WaveConstraint(K, WAVE_K)` specifies
    that we want distribute the K dimension among multiple waves which
    each wave operating on a tile size of WAVE_K. The assumption is
    that the K dimension has already been distributed among workgroups.
    If the K dimension has been distributed among workgroups with a
    tile size of BLOCK_K, then the number of waves along the K dimension
    is given by BLOCK_K // WAVE_K.

    This constraint adds an index constraint to the K-th dimension of a
    a tensor of the form WAVE_K * wave_id. The index of the wave
    is determined by the following mapping:
    workgroup id 0 -> wave/thread id x
    workgroup id 1 -> wave/thread id y
    workgroup id 2 -> wave/thread id z
    (If the tensor dimension has been distributed along workgroup dimension
    {0, 1, 2}, then the corresponding thread id is {x, y, z}).

    Because we represent the number of threads per block as
    [wave_id_0 * threads_per_wave, wave_id_1, wave_id_2], special care is
    required when computing wave_id_0. Specifically,
    wave_id_0 = floor(thread_id_0 / threads_per_wave)
    wave_id_1 = thread_id_1
    wave_id_2 = thread_id_2
    """

    dim: IndexExpr
    tile_size: IndexExpr
    wave_id: Optional[IndexExpr] = None

    def apply(self) -> IndexSequence:
        if self.wave_id is None:
            raise ValueError("Index is being computed without setting wave id")
        return IndexSequence(self.tile_size * self.wave_id, 1)
