# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from ..constraints import Constraint
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, get_custom, CustomOp
from .modulo_scheduling import ModuloScheduler
from .graph_utils import create_scheduling_edges, Edge
from .resources import get_available_resources, annotate_resource_usage
from ..visualization import visualize_edges, visualize_graph, visualize_schedule
from .loop_reconstruction import construct_pipelined_loop
from ..utils import (
    graph_copy,
    erase_graph,
    get_tiling_constraint,
    subs_idxc,
    get_assumptions,
    evaluate_with_assumptions,
)
import torch.fx as fx
from ....support.logging import get_logger

logger = get_logger("turbine.wave.scheduling.schedule")


"""
Formatting for different scheduling strategies:
Values: 0xAB where:
* A = Strategy types:
  * 0 = None
  * 1 = Solver Based
  * 2 = Heuristic Based (to come)
* B enumerates different strategy that share the same 0xA* bits.
"""


class SchedulingType(Enum):
    NONE = 0x00
    MODULO = 0x10


def is_solver_based(scheduling_type: SchedulingType):
    scheduling_ty_val = scheduling_type.value
    return scheduling_ty_val >= 0x10 and scheduling_ty_val < 0x20


def visualize_scheduling_graph(edges: list[Edge]):
    visualize_edges(edges, "reduction_graph.png")


def schedule_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
    scheduling_type: SchedulingType = SchedulingType.NONE,
):
    """
    Clones the reduction graph and does the following:
    1. Annotates resource usage for each node.
    2. Creates edges between outputs and return args for scheduling
       and assigns weights to all edges.
    Does scheduling on the cloned graph and applies the schedule
    to the original graph. Finally, erases the cloned graph.

    """
    if scheduling_type == SchedulingType.NONE:
        return {}
    reduction_graph = trace.get_subgraph(reduction.subgraph_name)
    graph, node_map = graph_copy(reduction_graph)

    if is_solver_based(scheduling_type):
        ignore_nodes, iter_args, output = annotate_resource_usage(graph)
        edges = create_scheduling_edges(graph, ignore_nodes, iter_args, output)
        scheduler = ModuloScheduler(graph, edges, get_available_resources())
        schedule, success = scheduler.schedule_graph()
        initiation_interval = scheduler.initiation_interval
        num_stages = scheduler.num_stages
    else:
        raise ValueError("Unknown scheduling type")

    if not success:
        raise ValueError("Scheduling failed.")

    visualize = False
    if visualize:
        if is_solver_based(scheduling_type):
            visualize_scheduling_graph(edges)
            visualize_graph(graph, "scheduling_fx_graph.png")
        visualize_schedule(schedule, initiation_interval, "schedule.html")

    # Apply schedule to original graph, specifying the stage
    # that each node is scheduled in as well as the cycle in
    # each stage when the node should be issued.
    inverse_node_map = {v: k for k, v in node_map.items()}
    iter_args: list[CustomOp] = []
    for node, cycle in schedule.items():
        if node not in inverse_node_map:
            continue
        custom = get_custom(inverse_node_map[node])
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % initiation_interval,
            "stage": cycle // initiation_interval,
            "initiation_interval": initiation_interval,
        }
        # Erase edges between outputs and iter args.
        if isinstance(get_custom(node), IterArg):
            node.args = ()
            iter_args.append(custom)

    for custom in iter_args:
        cycle = min([x.scheduling_parameters["absolute_cycle"] for x in custom.users])
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % initiation_interval,
            "stage": cycle // initiation_interval,
            "initiation_interval": initiation_interval,
        }

    erase_graph(graph)

    # After scheduling has completed, we have enough information to decide
    # whether to pipeline the loop. For pipelining to be possible, we need
    # to have atleast N iterations of the loop where N > num_stages - 1 (because
    # we will be peeling off num_stages iterations from the loop).
    tiling_constraint = get_tiling_constraint(reduction, constraints)
    max_induction_variable = subs_idxc(tiling_constraint.count)

    if max_induction_variable.is_number:
        # We can only do a compile-time check if the induction variable
        # is not dynamic.
        max_induction_variable = int(max_induction_variable)
        if max_induction_variable <= num_stages - 1:
            logger.warning(
                "Not enough iterations to pipeline the loop. Skipping pipelining."
            )
            return {}
    else:
        # Otherwise, we need to rely on assumptions provided by the author.
        assumptions = get_assumptions(constraints)
        if not assumptions:
            logger.warning(
                "No assumptions provided to determine if the loop can be pipelined. Skipping pipelining."
            )
            return {}

        result = evaluate_with_assumptions(
            constraints, max_induction_variable > num_stages - 1
        )
        if not result:
            logger.warning(
                "Not enough iterations to pipeline the loop. Skipping pipelining."
            )
            return {}

    new_reduction = construct_pipelined_loop(
        trace,
        reduction,
        reduction_graph,
        constraints,
        num_stages,
        initiation_interval,
        node_map,
        max_induction_variable,
        visualize,
        use_scheduling_barriers,
    )

    # Update new reduction count.
    new_reduction.count = max_induction_variable - (num_stages - 1)


def schedule_graph(
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
    scheduling_type: SchedulingType = SchedulingType.NONE,
):
    """
    Given a graph, pipelines the reductions in the graph.
    """

    if scheduling_type == SchedulingType.NONE:
        return

    def is_reduction(node: fx.Node) -> bool:
        return isinstance(get_custom(node), Reduction)

    reduction_nodes = trace.walk(is_reduction)
    if not reduction_nodes:
        return

    for reduction_node in reduction_nodes:
        schedule_reduction(
            get_custom(reduction_node),
            trace,
            constraints,
            use_scheduling_barriers,
            scheduling_type,
        )
