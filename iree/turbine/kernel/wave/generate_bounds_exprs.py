# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
import sympy
from ..ops.wave_ops import Read, Write
from .wave import CapturedTrace
from .constraints import Constraint, DistributionConstraint
from .utils.graph_utils import get_custom
from .utils.general_utils import find_index_bounds, get_hardware_constraint, is_shared_mem_access, remove_global_indexing
from .utils.symbol_utils import IndexSymbol, IndexExpr, subs_idxc, safe_subs

def _get_max_tile_size(
    dim: IndexSymbol,
    constraints: list[Constraint],
    vector_shapes: dict[IndexSymbol, int],
) -> IndexExpr:
    ret = sympy.sympify(vector_shapes[dim])
    for constraint in constraints:
        if isinstance(constraint, DistributionConstraint) and constraint.dim == dim:
            ret = sympy.Max(ret, constraint.tile_size)

    return ret

def generate_bounds_exprs(trace: CapturedTrace, constraints: list[Constraint]):
    """
    This pass generates bounds expressions for read and write ops.

    Bounds are used during MLIR lowering to handle partial access.
    """
    hardware_constraint = get_hardware_constraint(constraints)

    def is_read_write(node: fx.Node):
        return isinstance(get_custom(node), (Read, Write))

    nodes = trace.walk(is_read_write)
    for node in nodes:
        node = get_custom(node)
        vector_shapes = node.vector_shapes or hardware_constraint.vector_shapes
        is_shared_mem = is_shared_mem_access(node)
        bounds = find_index_bounds(constraints, node.index, vector_shapes)
        if is_shared_mem and bounds:
            bounds = remove_global_indexing(bounds, constraints)
            # Masking against global bounds was already handled when reading from
            # global mem, but we may still need to handle masking against vector
            # size during shared mem access.
            # Bound expression for this case will look like
            # `min(global_bound, vector_size)`.
            # Replace global bound with `max(tile_size, vector_size)` so the entire
            # expression `min(max(tile_size, vector_size), vector_size)` can be
            # simplified to just vector size.
            bounds = {
                k: safe_subs(
                    v, {k: _get_max_tile_size(k, constraints, vector_shapes)}
                )
                for k, v in bounds.items()
            }
            # Shared mem accesses are always access the full vector_shape tile,
            # so we can remove bounds that are divisible by vector size.
            bounds = {
                k: v
                for k, v in bounds.items()
                if subs_idxc(v % (vector_shapes[k] or 1)) != 0
            }

        if not bounds:
          continue

        node.update_arg("bounds", bounds)
