# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..wave.constraints import (
    Constraint,
    HardwareConstraint,
    WorkgroupConstraint,
    TilingConstraint,
)
from .._support.indexing import IndexingContext, IndexSequence, IndexSymbol, IndexExpr
from .._support.tracing import CapturedTrace
from ...support.logging import get_logger
from ..ops.wave_ops import get_custom, NewRegister, CustomOp, MMA, Reduction, ReduceOp
from .utils import get_hardware_vector_map
import torch.fx as fx

logger = get_logger("turbine.wave.register_analysis")


def set_register_shape(
    trace: CapturedTrace, custom: CustomOp, vector_map: dict[IndexSymbol, int]
) -> None:
    for custom_user in custom.users:
        if isinstance(custom_user, MMA):
            arg_index = custom_user.fx_node.args.index(custom.fx_node)
            get_thread_shape = lambda index: max(x.size for x in index.values())
            match arg_index:
                case 0:
                    custom.fx_node.thread_shape = get_thread_shape(
                        custom_user.lhs_index
                    )
                case 1:
                    custom.fx_node.thread_shape = get_thread_shape(
                        custom_user.rhs_index
                    )
                case 2:
                    custom.fx_node.thread_shape = get_thread_shape(
                        custom_user.acc_index
                    )
            break

        elif isinstance(custom_user, Reduction):
            idx = custom_user.init_args.index(custom.fx_node)
            iter_arg = get_custom(
                custom_user.iter_args(trace.get_subgraph(custom_user.subgraph_name))[
                    idx
                ]
            )
            set_register_shape(trace, iter_arg, vector_map)
            custom.fx_node.thread_shape = iter_arg.fx_node.thread_shape
            break
        elif isinstance(custom_user, ReduceOp):
            # Check that dim is non-reduction && in hw_constraint.vector_shape.
            is_parallel_dim = lambda dim: dim != custom_user.dim and dim in vector_map
            # TODO: Modify num_reduction_dims once we add support for multi-dim reduction.
            num_reduction_dims = 1
            register_shape = [
                vector_map[dim]
                for dim in custom_user.type.symbolic_shape
                if is_parallel_dim(dim)
            ]
            expected_result_rank = (
                len(custom_user.type.symbolic_shape) - custom_user.num_reduction_dims
            )
            # If rank do not match => some dims not found in hw_constraint.vector_shape.
            if len(register_shape) != expected_result_rank:
                raise NotImplementedError(
                    "NYI: Handling of dim not in vector_shapes during register analysis."
                )
            non_unit_dims = sum(1 for dim in register_shape if dim > 1)
            if non_unit_dims > 1:
                raise NotImplementedError(
                    "NYI: Currently Register semantic only support 0-D vector."
                )
            custom.fx_node.thread_shape = max(register_shape)
        else:
            raise NotImplementedError(
                f"Register shape propagation not implemented for {custom_user}"
            )


def determine_register_shape(
    trace: CapturedTrace | fx.Graph, constraints: list[Constraint]
) -> None:
    """
    Each register op is annotated with the wave shape of the register. This
    function determines the thread shape of the register based on the uses
    of the register in the graph.
    """
    register_nodes = trace.walk(lambda node: isinstance(get_custom(node), NewRegister))
    if not register_nodes:
        return
    vector_map = get_hardware_vector_map(constraints)
    for node in register_nodes:
        set_register_shape(trace, get_custom(node), vector_map)
