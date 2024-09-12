from ..constraints import Constraint, TilingConstraint
from ..._support.indexing import IndexSymbol
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, get_custom
from .modulo_scheduling import ModuloScheduler
from ..utils import graph_copy, erase_graph
import torch.fx as fx
import math
from collections import defaultdict, deque
import random


def liveness_analysis(
    graph: fx.Graph, constraints: list[Constraint], scheduler: ModuloScheduler
) -> dict[fx.Node, int]:
    """
    Perform liveness analysis on the graph to determine the live ranges of
    variables and use that to deduce how many rotating registers we need.
    """
    lifetime: dict[fx.Node, int] = {}
    for node in graph.nodes:
        custom = get_custom(node)
        if custom.scheduling_parameters is None:
            continue
        if node not in lifetime:
            lifetime[node] = 0
        for user in custom.users:
            if user.scheduling_parameters is None:
                continue
            lifetime[node] = max(
                user.scheduling_parameters["absolute_cycle"]
                - custom.scheduling_parameters["absolute_cycle"],
                lifetime[node],
            )

    # Determine how many copies we need for each node. If the lifetime of a node
    # is l clocks and the initiation interval is T, then only ceil(l/T) values
    # of the node can be live at the same time. We need to create copies of only
    # those nodes that are live at more than one stage.
    num_rotating_registers: dict[fx.Node, int] = {}
    for node, l in lifetime.items():
        n = math.ceil(l / scheduler.initiation_interval)
        if n > 1:
            num_rotating_registers[node] = n - 1

    return num_rotating_registers


def partition_graph_by_stage(
    graph: fx.Graph, scheduler: ModuloScheduler
) -> list[dict[int, list[fx.Node]]]:
    """
    Partition the graph into stages based on the scheduling parameters.
    """
    partitioned_graph: list[dict[int, list[fx.Node]]] = [
        defaultdict(list) for _ in range(scheduler.num_stages)
    ]
    for stage in range(scheduler.num_stages):
        for node in graph.nodes:
            custom = get_custom(node)
            if custom.scheduling_parameters is None:
                continue
            if custom.scheduling_parameters["stage"] == stage:
                cycle = custom.scheduling_parameters["cycle"]
                partitioned_graph[stage][cycle].append(node)
    return partitioned_graph


def interleave_instructions(instructions: list[tuple[int, int, fx.Node]]):
    """
    Interleave the instructions that are scheduled in the same cycle.
    Currently, we just randomly shuffle them, but we could also sort
    them based on some criteria.
    """
    rng = random.Random(0)
    rng.shuffle(instructions)


def add_nodes_by_schedule(
    reduction_graph: fx.Graph,
    partitioned_graph: list[dict[int, fx.Node]],
    partitioned_argument_map: list[dict[fx.Node, fx.Node]],
    stages: list[int],
    initiation_interval: int,
    induction_variable: IndexSymbol,
    current_induction_variables: list[int],
    rotating_registers: dict[fx.Node, list[fx.Node]],
):
    """
    Interleave the instructions in the partitioned graph by stage
    for a single initiation interval, updating the argument maps
    per stage starting at the provided start times and indices.
    """
    for cycle in range(initiation_interval):
        # Interleave the instructions that are scheduled at the same cycle.
        interleaved_instructions = []
        for iteration, stage in enumerate(stages):
            if stage is None:
                continue
            if cycle not in partitioned_graph[stage]:
                continue
            for node in partitioned_graph[stage][cycle]:
                interleaved_instructions.append((iteration, stage, node))
        interleave_instructions(interleaved_instructions)

        for iteration, stage, node in interleaved_instructions:
            node = get_custom(node)
            new_node = node.copy(
                new_graph=reduction_graph,
                arg_transform=lambda x: (
                    partitioned_argument_map[stage][x]
                    if x in partitioned_argument_map[stage]
                    else x
                ),
            )
            # Update the argument map for the current stage.
            partitioned_argument_map[stage][node] = new_node
            # Set the index for the new node by substituting the induction variable
            # for the current iteration.
            new_node.index = node.index
            for dim in new_node.index:
                new_node.index[dim] = new_node.index[dim].subs(
                    {induction_variable: current_induction_variables[iteration]}
                )
            # Update the rotating registers for the current node (if applicable).
            if node.fx_node in rotating_registers:
                rotating_registers[node.fx_node].append(new_node.fx_node)
                rotating_registers[node.fx_node].popleft()


def create_fill_stage_schedule(scheduler: ModuloScheduler) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the prologue (fill).
    """
    schedule = [
        [None for _ in range(scheduler.num_stages)]
        for _ in range(scheduler.num_stages - 1)
    ]
    for i in range(scheduler.num_stages - 1):
        schedule[i][i] = 0
        for k in range(i):
            schedule[i][k] = schedule[i - 1][k] + 1
    return schedule


def construct_prologue(
    reduction: Reduction,
    partitioned_graph: list[dict[int, fx.Node]],
    scheduler: ModuloScheduler,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
    stages: list[int],
):
    """
    Construct the prologue of the pipelined loop.
    For this, we need to copy nodes from the reduction_graph and insert them
    before the reduction operator in the root graph in the appropriate order.
    We also need to initialize the rotating registers and update the indices
    of the nodes to use the appropriate values of the induction variable.
    """
    partitioned_argument_map: list[dict[fx.Node, fx.Node]] = [
        {} for _ in range(scheduler.num_stages)
    ]
    with reduction.graph.inserting_before(reduction.fx_node):
        for i in range(scheduler.num_stages - 1):
            add_nodes_by_schedule(
                reduction.graph,
                partitioned_graph,
                partitioned_argument_map,
                stages[i],
                scheduler.initiation_interval,
                induction_variable,
                new_induction_variables,
                rotating_registers,
            )


def construct_kernel(
    reduction: Reduction,
    partitioned_graph: list[dict[int, fx.Node]],
    scheduler: ModuloScheduler,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
):
    """
    Construct the kernel of the pipelined loop.
    First, we construct a new reduction op with an empty graph.
    """
    flattened_rotating_registers = []
    for registers in rotating_registers.values():
        flattened_rotating_registers.extend(registers)
    with reduction.graph.inserting_before(reduction.fx_node):
        new_reduction = Reduction(
            reduction.axis,
            init_args=reduction.init_args + flattened_rotating_registers,
            subgraph_name="pipelined_reduction",
            implicit_captures=reduction.implicit_captures,
        ).add_to_graph(reduction.graph)
        pipelined_reduction = fx.Graph()
        reduction.graph.subgraphs["pipelined_reduction"] = pipelined_reduction
        partitioned_argument_map: list[dict[fx.Node, fx.Node]] = [
            {} for _ in range(scheduler.num_stages)
        ]
        add_nodes_by_schedule(
            pipelined_reduction,
            partitioned_graph,
            partitioned_argument_map,
            list(reversed(range(scheduler.num_stages))),
            scheduler.initiation_interval,
            induction_variable,
            new_induction_variables,
            rotating_registers,
        )


def get_induction_variable(
    reduction: Reduction, constraints: list[Constraint]
) -> IndexSymbol:
    induction_var = None
    for constraint in constraints:
        if (
            isinstance(constraint, TilingConstraint)
            and reduction.axis == constraint.dim
        ):
            induction_var = constraint.induction_var
            break
    else:
        raise ValueError(f"Could not find induction variable for reduction {reduction}")
    return induction_var


def construct_pipelined_loop(
    reduction: Reduction,
    graph: fx.Graph,
    constraints: list[Constraint],
    scheduler: ModuloScheduler,
):
    """
    Given a graph annotated with scheduling parameters, construct a pipelined loop
    with a prologue, kernel and epilogue.
    """
    induction_variable = get_induction_variable(reduction, constraints)
    num_rotating_registers = liveness_analysis(graph, constraints, scheduler)
    rotating_registers: dict[fx.Node, deque[fx.Node]] = {
        k: deque([None for _ in range(v)]) for k, v in num_rotating_registers.items()
    }
    partitioned_graph = partition_graph_by_stage(graph, scheduler)
    # Construct prologue.
    construct_prologue(
        reduction,
        partitioned_graph,
        scheduler,
        rotating_registers,
        induction_variable,
        list(range(scheduler.num_stages)),
        create_fill_stage_schedule(scheduler),
    )
    # Construct kernel.
    construct_kernel(
        reduction,
        partitioned_graph,
        scheduler,
        rotating_registers,
        induction_variable,
        [induction_variable + i for i in range(scheduler.num_stages)],
    )

    # Construct epilogue.
    # with reduction.graph.inserting_after(reduction.fx_node):
    #    construct_pipeline_stage(
    #        reduction,
    #        partitioned_graph,
    #        partitioned_argument_map,
    #        scheduler,
    #        rotating_registers,
    #        induction_variable,
    #        list(range(scheduler.num_stages)), # Change to last few iterations
    #        create_drain_stage_schedule(scheduler),
    #    )
    breakpoint()
