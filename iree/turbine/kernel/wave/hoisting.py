# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .constraints import Constraint
from .utils import get_induction_variable
from ...support.logging import get_logger
from iree.turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import *
from ..lang.global_symbols import *

logger = get_logger("turbine.wave.hoisting")


def get_hoistable_ops(
    graph: fx.Graph,
    captured_vars: list[CustomOp],
    induction_variable: IndexExpr,
) -> list[CustomOp]:
    """
    Get hoistable ops. Currently only handle allocs and read who doesn't depends on
    induction variables.

    Note: For codegen to work properly, we'd need to hoist allocs first. This is to avoid
    using alloc before defined/non-dominating behavior.
    (e.g hoisting read from global to shared before shared alloc is defined.)
    """
    hoistable_allocs = []
    hoistable_ops = []
    for node in graph.nodes:
        custom_node = get_custom(node)
        if isinstance(custom_node, Allocate):
            hoistable_allocs.append(custom_node)
        elif isinstance(custom_node, Read):
            if custom_node.index is None:
                continue
            # Only handle case where memory is captured var.
            # i.e it has source value from root graph.
            if not custom_node.memory in captured_vars:
                continue
            # Only handle case where we are not writing to the same memory.
            # Counterproof: we may expect different read if we write to same memory.
            if any(
                isinstance(get_custom(mem_user), Write)
                for mem_user in custom_node.memory.users
            ):
                continue
            # Only hoist Read that is loop invariant.
            if any(
                ind.start.has(induction_variable) for ind in custom_node.index.values()
            ):
                continue
            hoistable_ops.append(custom_node)
        else:
            continue
    all_hoistables_ops = hoistable_allocs + hoistable_ops
    return all_hoistables_ops


def remove_unused_captured_vars(reduction: CustomOp, subgraph: fx.Graph):
    captured_vars = reduction.captured_vars(subgraph)
    new_implicit_captures = list(reduction.implicit_captures)
    for captured_idx in reversed(range(len(captured_vars))):
        if len(captured_vars[captured_idx].users) == 0:
            get_custom(captured_vars[captured_idx]).erase()
            new_implicit_captures.pop(captured_idx)
            reduction.update_arg("implicit_captures", new_implicit_captures)


def hoist_loop_invariant_ops(trace: CapturedTrace, constraints: list[Constraint]):
    """Hoists ops that are loop-invariant from reduction subgraphs to outer root graph."""
    root_graph = trace.get_root_graph()
    for node in root_graph.nodes:
        custom_node = get_custom(node)
        match custom_node:
            case Reduction():
                with root_graph.inserting_before(custom_node.fx_node):
                    induction_variable = get_induction_variable(
                        custom_node, constraints
                    )
                    subgraph = trace.get_subgraph(custom_node.subgraph_name)
                    # Capture/root variables from outside the loop.
                    implicit_captures = custom_node.implicit_captures
                    # Captured variables from inside the loop.
                    captured_vars = custom_node.captured_vars(subgraph)
                    hoistable_ops = get_hoistable_ops(
                        subgraph, captured_vars, induction_variable
                    )
                    for hoistable_op in hoistable_ops:
                        new_op = hoistable_op.copy(new_graph=root_graph)
                        hoistable_op.replace_all_uses_with(new_op)
                        hoistable_op.erase()
                        if isinstance(hoistable_op, Read):
                            capture_arg = captured_vars.index(hoistable_op.memory)
                            new_op.update_arg("memory", implicit_captures[capture_arg])
                    # Clear/Remove unused captured var to correct codegen. Ops inside
                    # scf.for will be indexing/loading from the wrong bindings otherwise.
                    remove_unused_captured_vars(custom_node, subgraph)
