from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence
import shark_turbine.kernel.lang as tkl
from sympy import Expr, Symbol

from shark_turbine.kernel.ops.wave_ops import MMA
from .._support.indexing import IndexExpr, IndexSymbol


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
    def apply(self) -> IndexExpr:
        """Apply the constraint and get the resulting index expression."""
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
    """

    mma_type: MMAType
    threads_per_wave: int
    waves_per_block: Optional[Sequence[int]]

    def mma_matrix_shapes(self):
        # TODO: Eventually the shapes and indices should be provided by a tool
        match self.mma_type:
            case MMAType.F32_16x16x16_F16:
                return (16, 16, 16)
            case MMAType.F32_32x32x8_F16:
                return (32, 32, 8)

    def apply(self) -> IndexExpr:
        raise NotImplementedError("Not yet implemented")


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

    def apply(self) -> IndexExpr:
        match self.workgroup_dim:
            case 0:
                wg_dim = sym.WG0
            case 1:
                wg_dim = sym.WG1
            case _:
                raise ValueError("Invalid workgroup dimension. Expected 0 or 1.")
        return wg_dim * self.tile_size


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
