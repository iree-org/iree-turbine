# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from sympy import ceiling, Piecewise, floor

from .._support.indexing import IndexExpr, IndexSymbol, IndexSequence
from .._support.dtype import DataType
from ..lang.global_symbols import *


class MMAType(Enum):
    F32_16x16x16_F16 = 0
    F32_32x32x8_F16 = 1
    F32_16x16x32_F8 = 2
    F32_32x32x16_F8 = 3


class MMAOperand(Enum):
    M = 0
    N = 1
    K = 2


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

    Both mma constraints and vector shapes can be specified, but
    the mapping from symbols to shapes should be injective.
    """

    threads_per_wave: int
    waves_per_block: Optional[tuple[int, int, int]] = None
    mma_type: Optional[MMAType] = MMAType.F32_16x16x16_F16
    vector_shapes: Optional[dict[IndexSymbol, int]] = None
    max_bits_per_load: int = 128

    def max_elems_per_load(self, element_type: DataType) -> int:
        return self.max_bits_per_load // element_type.bitwidth()

    def get_thread_id_from_workgroup_dim(self, workgroup_dim: int) -> IndexSymbol:
        match workgroup_dim:
            case 0:
                return THREAD_0
            case 1:
                return THREAD_1
            case 2:
                return THREAD_2
            case _:
                raise ValueError("Invalid workgroup dimension. Expected 0, 1 or 2.")

    @property
    def mma_matrix_shapes(self) -> tuple[int]:
        # TODO: Eventually the shapes and indices should be provided by a tool
        match self.mma_type:
            case MMAType.F32_16x16x16_F16:
                return (16, 16, 16)
            case MMAType.F32_32x32x8_F16:
                return (32, 32, 8)
            case MMAType.F32_16x16x32_F8:
                return (16, 16, 32)
            case MMAType.F32_32x32x16_F8:
                return (32, 32, 16)
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

    # Inline substitution for vector_size given index map. In the future we can add support for other members.
    def subs_vector_shapes(self, index_map: dict[IndexSymbol, int]):
        if self.vector_shapes is None:
            return
        for vector_dim, vector_size in self.vector_shapes.items():
            if isinstance(vector_size, IndexExpr):
                self.vector_shapes[vector_dim] = vector_size.subs(index_map)

    def compute_access_pattern_using_vector_shapes(
        self,
        dim: IndexSymbol,
        workgroup_dim: int,
        elements_per_thread: int | IndexSymbol,
        stride: int,
    ) -> IndexSequence:
        thread_id = self.get_thread_id_from_workgroup_dim(workgroup_dim)
        return IndexSequence(
            thread_id * elements_per_thread, elements_per_thread, stride
        )

    def apply(
        self,
        dim: IndexSymbol,
        constraint_index: int | MMAOperand,
        elements_per_thread: int | IndexSymbol,
        stride: int,
        is_mma_dim: bool,
    ) -> IndexSequence:
        if not is_mma_dim:
            return self.compute_access_pattern_using_vector_shapes(
                dim, constraint_index, elements_per_thread, stride
            )
        lane = self.linearized_thread_id % self.threads_per_wave
        match self.mma_type:
            # (M x K, N x K) -> M x N
            case MMAType.F32_16x16x16_F16:
                offset = [
                    Piecewise(
                        (lane % 16, ~MMA_ACC),
                        (4 * floor(lane / 16), MMA_ACC),
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
            case MMAType.F32_32x32x8_F16:
                offset = [
                    Piecewise(
                        (lane % 32, ~MMA_ACC),
                        (
                            (8 * floor(GPR_NUM / 4) % 32)
                            + 4 * floor(lane / 32)
                            + (GPR_NUM % 4),
                            MMA_ACC,
                        ),
                    ),  # M
                    lane % 32,  # N
                    4 * floor(lane / 32),  # K
                ]
                size = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    4,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (32, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case MMAType.F32_16x16x32_F8:
                offset = [
                    Piecewise(
                        (lane % 16, ~MMA_ACC), (4 * floor(lane / 16), MMA_ACC)
                    ),  # M
                    lane % 16,  # N
                    8 * floor(lane / 16),  # K
                ]
                size = [
                    Piecewise((1, ~MMA_ACC), (4, MMA_ACC)),  # M
                    1,  # N
                    8,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case MMAType.F32_32x32x16_F8:
                offset = [
                    Piecewise(
                        (lane % 32, ~MMA_ACC),
                        (
                            (8 * floor(GPR_NUM / 4) % 32)
                            + 4 * floor(lane / 32)
                            + (GPR_NUM % 4),
                            MMA_ACC,
                        ),
                    ),  # M
                    lane % 32,  # N
                    8 * floor(lane / 32),  # K
                ]
                size = [
                    Piecewise((1, ~MMA_ACC), (16, MMA_ACC)),  # M
                    1,  # N
                    8,  # K
                ]
                stride = [
                    Piecewise((1, ~MMA_ACC), (32, MMA_ACC)),  # M
                    1,  # N
                    1,  # K
                ]
            case _:
                raise ValueError("Unsupported MMA type")
        assert isinstance(
            constraint_index, MMAOperand
        ), f"Invalid MMA operand {constraint_index}"
        return IndexSequence(
            offset[constraint_index.value],
            size[constraint_index.value],
            stride[constraint_index.value],
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

    @property
    def count(self) -> IndexExpr:
        """
        Returns an expression for the total number of workgroups for the specific workgroup_dim.
        """
        return ceiling(self.dim / self.tile_size)

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
    grid: list[IndexExpr] = [constraint.count for constraint in wg_constraints]
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

    @property
    def count(self) -> IndexExpr:
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


def get_constrained_shape(
    shape: list[IndexExpr], constraints: list[WorkgroupConstraint | TilingConstraint]
) -> tuple[IndexExpr]:
    """
    Given a shape, workgroup and tiling constraints, returns the shape
    of the distributed and tiled tensor.
    """
    constrained_shape = list(shape)
    for i, dim in enumerate(shape):
        for constraint in constraints:
            if isinstance(constraint, WorkgroupConstraint) or isinstance(
                constraint, TilingConstraint
            ):
                if dim == constraint.dim:
                    constrained_shape[i] = constraint.tile_size
    return tuple(constrained_shape)
