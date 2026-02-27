# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Analysis for single-writer memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TYPE_CHECKING
import sympy

if TYPE_CHECKING:
    from ..lang.tkw_types import IndexMapping
    from .._support.indexing import IndexExpr


class AnalysisOutcome(Enum):
    """Result of analyzing a write operation for single-writer property."""
    PROVEN_UNIQUE = auto()
    OWNER_PREDICATE = auto()
    NEEDS_GUARD = auto()


@dataclass(frozen=True, slots=True)
class OwnerPredicate:
    """Structured owner predicate: grid[axis] == value."""
    axis: int
    value: int = 0


@dataclass(slots=True)
class AnalysisResult:
    """Analysis result with outcome and optional owner predicate."""
    outcome: AnalysisOutcome
    predicate: Optional[OwnerPredicate] = None

    @classmethod
    def unique(cls) -> AnalysisResult:
        return cls(AnalysisOutcome.PROVEN_UNIQUE)

    @classmethod
    def owner(cls, axis: int, value: int = 0) -> AnalysisResult:
        return cls(AnalysisOutcome.OWNER_PREDICATE, OwnerPredicate(axis, value))

    @classmethod
    def guard(cls) -> AnalysisResult:
        return cls(AnalysisOutcome.NEEDS_GUARD)


def _is_constant(expr) -> bool:
    """Check if expression is a constant value."""
    return isinstance(expr, (int, sympy.Integer)) or getattr(expr, 'is_number', False)


def analyze_write(
    mapping: Optional[IndexMapping],
    ref_shape: tuple[IndexExpr, ...],
    has_identity: bool = False,
) -> AnalysisResult:
    """Analyze write for single-writer property."""
    # Fast path: identity mapping = each thread writes unique location
    if has_identity or mapping is None:
        return AnalysisResult.unique()
    
    # Check identity via mapping API (reuses existing infrastructure)
    if mapping.is_identity():
        return AnalysisResult.unique()
    
    # Attempt owner predicate extraction for reduction patterns
    pred = _extract_owner_predicate(mapping)
    return AnalysisResult.owner(pred.axis, pred.value) if pred else AnalysisResult.guard()


def _extract_owner_predicate(mapping: IndexMapping) -> Optional[OwnerPredicate]:
    """Extract canonical owner predicate from non-identity mapping."""
    # Pattern: All output dimensions are constants (broadcast/reduction)
    if all(_is_constant(expr) for expr in mapping.output_mapping.values()):
        return OwnerPredicate(axis=0, value=0)
    
    # Pattern: Floor division creates many-to-one mapping - needs full guard
    for expr in mapping.output_mapping.values():
        if isinstance(expr, sympy.Expr) and expr.has(sympy.floor):
            return None
    
    return None
