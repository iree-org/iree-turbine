# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Code generation for guarded memory writes."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from .ir import (
    IndexType,
    IntegerType,
    MemRefType,
    arith_d,
    memref_d,
    scf_d,
    InsertionPoint,
    Attribute,
)
from .write_analysis import AnalysisResult, AnalysisOutcome, OwnerPredicate
from .base import options

if TYPE_CHECKING:
    from .vector_codegen import ThreadEmitter


def emit_guarded_store(
    emitter: ThreadEmitter,
    analysis: AnalysisResult,
    store_fn: Callable[[], None],
) -> None:
    """Emit store with appropriate guard based on analysis result."""
    if not options.enable_single_writer_guards:
        store_fn()
        return

    match analysis.outcome:
        case AnalysisOutcome.PROVEN_UNIQUE:
            store_fn()
        case AnalysisOutcome.OWNER_PREDICATE:
            _emit_owner_guard(emitter, analysis.predicate, store_fn)
        case AnalysisOutcome.NEEDS_GUARD:
            _emit_atomic_guard(emitter, store_fn)


def _emit_owner_guard(
    emitter: ThreadEmitter,
    predicate: OwnerPredicate,
    store_fn: Callable[[], None],
) -> None:
    """Emit scf.if guard: execute store only if predicate holds."""
    axis_val = emitter.lookup_grid_axis_value(predicate.axis).ir_value
    const_val = arith_d.constant(IndexType.get(), predicate.value)
    cond = arith_d.cmpi(arith_d.CmpIPredicate.eq, axis_val, const_val)
    
    if_op = scf_d.IfOp(cond, results_=[])
    with InsertionPoint(if_op.then_block):
        store_fn()
        scf_d.yield_([])


def _emit_atomic_guard(
    emitter: ThreadEmitter,
    store_fn: Callable[[], None],
) -> None:
    """Emit atomic test-and-set guard for first-writer-wins semantics.
    
    Uses atomic_rmw to ensure only one thread (the first to arrive) executes store.
    The flag must be pre-allocated in workgroup memory and initialized before kernel.
    """
    i32 = IntegerType.get_signless(32)
    idx_ty = IndexType.get()
    
    # Workgroup-local flag (address space 3) - must be pre-initialized
    flag_type = MemRefType.get([1], i32, memory_space=Attribute.parse("3"))
    flag = memref_d.alloca(flag_type, [], [])
    
    c0 = arith_d.constant(idx_ty, 0)
    c1_i32 = arith_d.constant(i32, 1)
    c0_i32 = arith_d.constant(i32, 0)
    
    # Atomic add 1, returns old value - first thread gets 0
    old_val = memref_d.atomic_rmw(arith_d.AtomicRMWKind.addi, c1_i32, flag, [c0])
    is_first = arith_d.cmpi(arith_d.CmpIPredicate.eq, old_val, c0_i32)
    
    if_op = scf_d.IfOp(is_first, results_=[])
    with InsertionPoint(if_op.then_block):
        store_fn()
        scf_d.yield_([])

