from ..constraints import Constraint, TilingConstraint
from ..._support.indexing import IndexSymbol
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import Reduction, IterArg, Output, Write, GetResult, get_custom
from .modulo_scheduling import ModuloScheduler
from ..utils import graph_copy, erase_graph
from ..utils import subs_idxc
import torch.fx as fx
import math
from collections import defaultdict, deque, ChainMap
from ..visualization import visualize_mapped_graphs
from ....support.logging import get_logger
from ...lang.global_symbols import SHARED_ADDRESS_SPACE
import random
from typing import Optional
from .loop_reconstruction_utils import ArgumentContext

logger = get_logger("turbine.wave.scheduling.loop_reconstruction")


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
            logger.debug(
                f"Node: {node}, User: {user.fx_node}, lifetime: {user.scheduling_parameters['stage'] - custom.scheduling_parameters['stage']}"
            )
            lifetime[node] = max(
                user.scheduling_parameters["stage"]
                - custom.scheduling_parameters["stage"],
                lifetime[node],
            )

    # Determine how many copies we need for each node. If the lifetime of a node
    # is l clocks and the initiation interval is T, then only ceil(l/T) values
    # of the node can be live at the same time. We need to create copies of only
    # those nodes that are live at more than one stage.
    num_rotating_registers: dict[fx.Node, int] = {}
    for node, l in lifetime.items():
        if node in num_rotating_registers:
            continue
        custom = get_custom(node)
        if (
            isinstance(custom, Write)
            and custom.memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            continue
        if l > 0:
            num_rotating_registers[node] = l

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
    arg_context: ArgumentContext,
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
        logger.debug(f"Cycle: {cycle}")
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
            custom_node = get_custom(node)
            for arg in node.args:
                if (stage, arg) in arg_context:
                    logger.debug(
                        f"Found arg: {arg} in partitioned argument map. Using {arg_context.query_arg(stage, arg)}."
                    )
                    continue
            new_node = custom_node.copy(
                new_graph=reduction_graph,
                arg_transform=lambda x: (
                    arg_context.query_arg(stage, x) if (stage, x) in arg_context else x
                ),
            )
            # Update the argument map for the current stage.
            arg_context.map_arg(stage, node, new_node.fx_node)
            logger.debug(
                f"Copying Node: {node}, Stage: {stage}, Iteration: {iteration} -> {new_node.fx_node}"
            )
            # Set the index for the new node by substituting the induction variable
            # for the current iteration.
            new_node.index = node.index
            for dim in new_node.index:
                new_node.index[dim] = new_node.index[dim].subs(
                    {induction_variable: current_induction_variables[iteration]}
                )
            # Add scheduling parameters for debugging.
            new_node.scheduling_parameters = node.scheduling_parameters
            # Update the rotating registers for the current node (if applicable).
            if node in rotating_registers:
                rotating_registers[node].append(new_node.fx_node)
                rotating_registers[node].popleft()

            # Update the argument map for the next stage.


def create_fill_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the prologue (fill).
    This looks like:
    [0 None None None]
    [1    0 None None]
    [2    1    0 None]
    """
    schedule = []
    for i in range(n - 1):
        row = list(range(i, -1, -1))
        row.extend([None] * (n - i - 1))
        schedule.append(row)
    return schedule


def create_drain_stage_schedule(n: int) -> list[list[int]]:
    """
    Create the schedule of which stages need to be interleaved for the epilogue (drain).
    This looks like:
    [None    3    2 1]
    [None None    3 2]
    [None None None 3]
    """
    schedule = []
    for i in range(n - 1):
        row = [None] * (i + 1)
        row.extend(range(n - 1, i, -1))
        schedule.append(row)
    return schedule


def construct_prologue(
    reduction_subgraph: fx.Graph,
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
    arg_context = ArgumentContext(reduction_subgraph, scheduler.num_stages)
    with reduction.graph.inserting_before(reduction.fx_node):
        for i in range(scheduler.num_stages - 1):
            add_nodes_by_schedule(
                reduction.graph,
                partitioned_graph,
                arg_context,
                stages[i],
                scheduler.initiation_interval,
                induction_variable,
                new_induction_variables,
                rotating_registers,
            )


def flatten_dict_values(
    rotating_registers: dict[fx.Node, list[fx.Node]]
) -> list[fx.Node]:
    """
    Flatten the values of the rotating registers into a list.
    """
    return [
        register for registers in rotating_registers.values() for register in registers
    ]


def unflatten_dict_values(
    rotating_registers_shapes: dict[fx.Node, int], values: list[fx.Node]
) -> dict[fx.Node, list[fx.Node]]:
    """
    Unflatten the values of the rotating registers into a dictionary
    using the provided shapes.
    """
    rotating_registers = {}
    count = 0
    for node, shape in rotating_registers_shapes.items():
        rotating_registers[node] = deque(values[count : count + shape])
        count += shape
    return rotating_registers


def push_rotating_registers(
    arg_context: ArgumentContext,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    graph: fx.Graph,
    node_map: dict[fx.Node, fx.Node],
    create_new_nodes: bool = False,
) -> dict[fx.Node, deque[fx.Node]]:
    """
    Pushes the rotating registers into the argument map
    at the appropriate stages. Create new nodes in the
    specified graph if requested.

    For each rotating register,
    we evaluate which stage it belongs to an update the argument
    map for the next stage and n - 1 stages after it, where
    n is the total number of rotating registers.
    """
    new_rotating_registers: dict[fx.Node, deque[fx.Node]] = {}
    count = 0
    for node, registers in rotating_registers.items():
        new_registers: deque[fx.Node] = deque()
        for i, register in enumerate(registers):
            custom = get_custom(node)
            stage = custom.scheduling_parameters["stage"]
            mapped_stage = stage + len(registers) - i
            if create_new_nodes:
                iter_arg = IterArg(f"rotating_reg_{count}").add_to_graph(graph)
                arg_context.map_arg(mapped_stage, node, iter_arg)
                new_registers.append(iter_arg)
                logger.debug(
                    f"Mapped orig: {node_map[node]} / mapped: {iter_arg} to stage {mapped_stage}."
                )
            else:
                arg_context.map_arg(mapped_stage, node, register)
            count += 1
        if new_registers:
            new_rotating_registers[node] = new_registers
    return new_rotating_registers


def construct_kernel(
    reduction_subgraph: fx.Graph,
    reduction: Reduction,
    partitioned_graph: list[dict[int, fx.Node]],
    scheduler: ModuloScheduler,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
    node_map: dict[fx.Node, fx.Node],
) -> tuple[Reduction, fx.Graph]:
    """
    Construct the kernel of the pipelined loop.
    First, we construct a new reduction op with an empty graph.
    Then, we set the init args, construct the iter args and add the ops.
    Finally, we create the output node with the return values.
    The iter args/results of the pipelined reduction are always:
    [results0, result1, ..., resultN, rotating_reg0, rotating_reg1, ..., rotating_regN]
    """

    flattened_rotating_registers = flatten_dict_values(rotating_registers)

    reduction_init_args = reduction.init_args
    reduction_iter_args = reduction.iter_args(reduction_subgraph)

    with reduction.graph.inserting_before(reduction.fx_node):
        pipelined_reduction = Reduction(
            reduction.axis,
            init_args=reduction_init_args + flattened_rotating_registers,
            subgraph_name="pipelined_reduction",
            implicit_captures=reduction.implicit_captures,
        ).add_to_graph(reduction.graph)
        custom = get_custom(pipelined_reduction)
        pipelined_reduction.index = reduction.index
        pipelined_reduction_graph = fx.Graph()
        reduction.graph.subgraphs["pipelined_reduction"] = pipelined_reduction_graph

        # Update the argument map for the new reduction.
        arg_context = ArgumentContext(reduction_subgraph, scheduler.num_stages)

        # For the original iter args, we just map the old ones to the new ones.
        # Do this for all stages, since the original iter args are "dummy" nodes
        # during scheduling.
        for node in reduction_iter_args:
            custom = get_custom(node)
            iter_arg = IterArg(node.name).add_to_graph(pipelined_reduction_graph)
            arg_context.map_arg_all_stages(node, iter_arg)

        # Push the rotating registers into the argument map.
        new_rotating_registers: dict[fx.Node, deque[fx.Node]] = push_rotating_registers(
            arg_context,
            rotating_registers,
            pipelined_reduction_graph,
            node_map,
            create_new_nodes=True,
        )

        add_nodes_by_schedule(
            pipelined_reduction_graph,
            partitioned_graph,
            arg_context,
            list(reversed(range(scheduler.num_stages))),
            scheduler.initiation_interval,
            induction_variable,
            new_induction_variables,
            new_rotating_registers,
        )

        # Create output node (last node in the graph).
        original_output = get_custom(reduction_subgraph._root.prev)
        return_vals = []
        for register in original_output.return_vals[0]:
            custom = get_custom(register)
            stage = custom.scheduling_parameters["stage"]
            return_vals.append(arg_context.query_arg(stage, register))
        for registers in new_rotating_registers.values():
            return_vals.extend(registers)

        Output(return_vals).add_to_graph(pipelined_reduction_graph)
        reduction.replace_all_uses_with(pipelined_reduction)

        visualize = False
        if visualize:
            visualize_mapped_graphs(
                pipelined_reduction_graph,
                arg_context.argument_map,
                "kernel.png",
            )

        return pipelined_reduction, pipelined_reduction_graph


def construct_epilogue(
    reduction_subgraph: fx.Graph,
    pipelined_reduction: Reduction,
    partitioned_graph: list[dict[int, fx.Node]],
    scheduler: ModuloScheduler,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
    stages: list[int],
    num_rotating_registers: dict[fx.Node, int],
):
    """
    Construct the epilogue of the pipelined loop.
    The difference from the prologue is that we need to map the results
    of the pipelined reduction to the remaining stages. (In the prologue,
    no iteration is every completed and so we don't compute the final results)
    We emit GetResult nodes for the rotating registers and map them to
    the different epilogue stages.
    """
    arg_context = ArgumentContext(reduction_subgraph, scheduler.num_stages)

    existing_get_results: list[GetResult] = [
        x for x in pipelined_reduction.users if isinstance(x, GetResult)
    ]
    existing_get_results = sorted(existing_get_results, key=lambda x: x.res_idx)

    with pipelined_reduction.graph.inserting_before(
        existing_get_results[0].fx_node.next
    ):
        # Add get result nodes for the rotating registers and update the
        # argument map with them.
        rotating_registers_get_results = []
        for i in range(len(existing_get_results), len(rotating_registers)):
            rotating_registers_get_results.append(
                GetResult(pipelined_reduction.fx_node, i).add_to_graph(
                    pipelined_reduction.graph
                )
            )
        rotating_registers = unflatten_dict_values(
            num_rotating_registers, rotating_registers_get_results
        )

        # Push the rotating registers onto the argument map.
        push_rotating_registers(arg_context, rotating_registers, None, None, False)

        for i in range(scheduler.num_stages - 1):
            add_nodes_by_schedule(
                pipelined_reduction.graph,
                partitioned_graph,
                arg_context,
                stages[i],
                scheduler.initiation_interval,
                induction_variable,
                new_induction_variables,
                rotating_registers,
            )

        # Replace the existing get results with the new results.
        new_results = arg_context.query_mapped_results()
        for i, get_result in enumerate(existing_get_results):
            get_result.replace_all_uses_with(new_results[i])


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
    node_map: dict[fx.Node, fx.Node],
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
    tiling_constraint = [
        c
        for c in constraints
        if isinstance(c, TilingConstraint) and c.dim == reduction.axis
    ][0]
    max_induction_variable = subs_idxc(tiling_constraint.dim) // subs_idxc(
        tiling_constraint.tile_size
    )
    # Construct prologue.
    construct_prologue(
        graph,
        reduction,
        partitioned_graph,
        scheduler,
        rotating_registers,
        induction_variable,
        list(range(scheduler.num_stages)),
        create_fill_stage_schedule(scheduler.num_stages),
    )
    # Construct kernel.
    pipelined_reduction, pipelined_reduction_graph = construct_kernel(
        graph,
        reduction,
        partitioned_graph,
        scheduler,
        rotating_registers,
        induction_variable,
        [induction_variable + i for i in range(scheduler.num_stages)],
        node_map,
    )
    # Construct epilogue.
    construct_epilogue(
        graph,
        get_custom(pipelined_reduction),
        partitioned_graph,
        scheduler,
        rotating_registers,
        induction_variable,
        [max_induction_variable - i for i in range(scheduler.num_stages)],
        create_drain_stage_schedule(scheduler.num_stages),
        num_rotating_registers,
    )
    breakpoint()
    # Remove the unpipelined reduction.
    reduction.graph.erase_node(reduction.fx_node)
