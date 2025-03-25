# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel._support.indexing import IndexExpr, IndexSymbol
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
from .utils.symbol_utils import subs_idxc
from .constraints import (
    Constraint,
    WorkgroupConstraint,
    WaveConstraint,
    TilingConstraint,
)


@dataclass
class SymbolicAlias:
    """
    A constraint of the form `tkw.SymbolicConstraint(K, SYMBOLIC_K)` specifies
    that the relationship between the source and target symbols is given by
    source = source_to_target(target).

    SymbolicAliases are modeled in the compiler as additional workgroup, wave,
    and tiling constraints that are derived from the source. They are ignored
    during expansion and utilize the same workgroup and wave ids as the
    target symbol.
    """

    source: IndexSymbol | IndexExpr
    target: IndexSymbol | IndexExpr
    source_to_target: Callable[[IndexSymbol | IndexExpr], IndexSymbol | IndexExpr]

    def apply(self, target: IndexSymbol | IndexExpr) -> IndexSymbol | IndexExpr:
        return subs_idxc(self.source_to_target(target))

    def create_new_constraints(self, constraints: list[Constraint]) -> list[Constraint]:
        """
        Creates new constraints for the given constraints with the appropriate
        substitution of the indexing context.

        """
        new_constraints = []
        if not constraints:
            return new_constraints
        match constraints[0]:
            case WorkgroupConstraint():
                build_constraint = lambda x, y, z: WorkgroupConstraint(x, y, z)
                id_fn = lambda x: x.workgroup_dim
            case WaveConstraint():
                build_constraint = lambda x, y, z: WaveConstraint(x, y, z)
                id_fn = lambda x: x.wave_id
            case TilingConstraint():
                build_constraint = lambda x, y, z: TilingConstraint(x, y, z)
                id_fn = lambda x: x.induction_var
        for constraint in constraints:
            if self.target == constraint.dim:
                tile_size = self.apply(constraint.tile_size)
                if tile_size.is_number and tile_size == 0:
                    continue
                new_constraints.append(
                    build_constraint(
                        self.source,
                        self.apply(constraint.tile_size),
                        id_fn(constraint),
                    )
                )
        return new_constraints
