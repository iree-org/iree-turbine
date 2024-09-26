# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..constraints import Constraint
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, get_custom
from .modulo_scheduling import ModuloScheduler
from .graph_utils import create_scheduling_edges, Edge
from .resources import get_available_resources, annotate_resource_usage
from ..visualization import visualize_edges, visualize_graph, visualize_schedule
from .loop_reconstruction import construct_pipelined_loop
from ..utils import graph_copy, erase_graph, get_tiling_constraint, subs_idxc
import torch.fx as fx
from ....support.logging import get_logger

logger = get_logger("turbine.wave.scheduling.schedule")


def visualize_scheduling_graph(edges: list[Edge]):
    visualize_edges(edges, "reduction_graph.png")


def schedule_reduction(
    reduction: Reduction,
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
):
    """
    Clones the reduction graph and does the following:
    1. Annotates resource usage for each node.
    2. Creates edges between outputs and return args for scheduling
       and assigns weights to all edges.
    Does scheduling on the cloned graph and applies the schedule
    to the original graph. Finally, erases the cloned graph.

    """
    reduction_graph = trace.get_subgraph(reduction.subgraph_name)
    graph, node_map = graph_copy(reduction_graph)
    ignore_nodes, iter_args, output = annotate_resource_usage(graph)
    edges = create_scheduling_edges(graph, ignore_nodes, iter_args, output)

    visualize = False
    if visualize:
        visualize_scheduling_graph(edges)
        visualize_graph(graph, "scheduling_fx_graph.png")

    scheduler = ModuloScheduler(graph, edges, get_available_resources())
    schedule, success = scheduler.schedule_graph()
    if not success:
        raise ValueError("Scheduling failed.")
    if visualize:
        visualize_schedule(schedule, scheduler.initiation_interval, "schedule.html")

    # Apply schedule to original graph, specifying the stage
    # that each node is scheduled in as well as the cycle in
    # each stage when the node should be issued.
    inverse_node_map = {v: k for k, v in node_map.items()}
    for node, cycle in schedule.items():
        if node not in inverse_node_map:
            continue
        custom = get_custom(inverse_node_map[node])
        custom.scheduling_parameters = {
            "absolute_cycle": cycle,
            "cycle": cycle % scheduler.initiation_interval,
            "stage": cycle // scheduler.initiation_interval,
            "initiation_interval": scheduler.initiation_interval,
        }
        # Erase edges between outputs and iter args.
        if isinstance(get_custom(node), IterArg):
            node.args = ()

    erase_graph(graph)

    # After scheduling has completed, we have enough information to decide
    # whether to pipeline the loop. For pipelining to be possible, we need
    # to have atleast N iterations of the loop where N > num_stages - 1 (because
    # we will be peeling off num_stages iterations from the loop).
    tiling_constraint = get_tiling_constraint(reduction, constraints)
    max_induction_variable = int(
        subs_idxc(tiling_constraint.dim) // subs_idxc(tiling_constraint.tile_size)
    )
    if max_induction_variable <= scheduler.num_stages - 1:
        logger.warn("Not enough iterations to pipeline the loop. Skipping pipelining.")
        return {}

    new_reduction = construct_pipelined_loop(
        trace,
        reduction,
        reduction_graph,
        constraints,
        scheduler,
        node_map,
        max_induction_variable,
        visualize,
        use_scheduling_barriers,
    )

    # Update new reduction count.
    new_reduction.count = max_induction_variable - (scheduler.num_stages - 1)


def schedule_graph(
    trace: CapturedTrace,
    constraints: list[Constraint],
    use_scheduling_barriers: bool = False,
):
    """
    Given a graph, pipelines the reductions in the graph.
    """

    def is_reduction(node: fx.Node) -> bool:
        return isinstance(get_custom(node), Reduction)

    reduction_nodes = trace.walk(is_reduction)
    if not reduction_nodes:
        return

    for reduction_node in reduction_nodes:
        schedule_reduction(
            get_custom(reduction_node), trace, constraints, use_scheduling_barriers
        )
