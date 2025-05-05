# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils.symbol_utils import subs_idxc
from .constraints import Constraint
from .utils.general_utils import get_tiling_constraint
from iree.turbine.kernel._support.tracing import CapturedTrace
import torch.fx as fx
from ..ops.wave_ops import Iterate, Output, Placeholder, get_custom


def remap_iter_args(
    iter_args: list[fx.Node], output: Output, value_use_map: dict[fx.Node, fx.Node]
) -> None:
    """
    Add a mapping of iter_args to the output return values in order to keep track
    of how uses need to be updated for unrolled iterations.
    """
    for iter_arg, return_val in zip(iter_args, output.return_vals[0]):
        value_use_map[iter_arg] = return_val


def unroll(
    iterate: Iterate,
    unroll_factor: int,
    trace: CapturedTrace,
    constraints: list[Constraint],
) -> None:
    """
    Unroll an iterate node in the graph `unroll_factor` times.
    This is done by creating `unroll_factor` copies of the iteration body
    and adjusting the step size and boundaries accordingly.
    """
    assert unroll_factor > 1, "Unroll factor must be greater than 1"

    if iterate.count is None:
        # This is required if the upper bound has not yet been statically
        # determined from the constraints, e.g. when unrolling is used before the scheduling pass.
        tiling_constraint = get_tiling_constraint(iterate, constraints)
        iterate.count = subs_idxc(tiling_constraint.count)
    if iterate.count / unroll_factor < 1:
        raise ValueError("Unroll factor is too large for the iteration count.")
    if iterate.count % unroll_factor != 0:
        raise ValueError("Unroll factor must divide the iteration count evenly.")

    iterate.count = iterate.count / unroll_factor
    iterate.step = iterate.step * unroll_factor

    graph = trace.get_subgraph(iterate.subgraph_name)

    # Keep track of the required remappings:
    # Cloned nodes that referred to the iter args need to be mapped to
    # the return values from iterate
    output = get_custom(list(graph.nodes)[-1])
    assert isinstance(output, Output)
    value_use_map: dict[fx.Node, fx.Node] = {}
    remap_iter_args(iterate.iter_args(graph), output, value_use_map)

    def value_mapper(old_arg: fx.Node) -> fx.Node:
        if old_arg in value_use_map:
            return value_use_map[old_arg]
        else:
            return old_arg

    original_body_nodes = list(graph.nodes)
    for unroll_idx in range(0, unroll_factor - 1):
        for node in original_body_nodes:
            original = get_custom(node)
            if isinstance(original, Placeholder):
                continue
            copy = original.copy(
                original.name + f"_unrolled{unroll_idx}",
                arg_transform=value_mapper,
                anchor=list(graph.nodes)[-2],
            )
            value_use_map[original.fx_node] = copy.fx_node

            if isinstance(copy, Output):
                remap_iter_args(iterate.iter_args(graph), copy, value_use_map)
                # Erase the original output node as the iterate must only
                # have one output at the end of the body.
                if node in value_use_map:
                    get_custom(value_use_map[node]).erase()
                else:
                    original.erase()
                value_use_map[node] = copy.fx_node
