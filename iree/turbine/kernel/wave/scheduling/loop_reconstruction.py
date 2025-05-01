from ..constraints import Constraint
from ..._support.indexing import IndexSymbol
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import (
    Iterate,
    IterArg,
    Placeholder,
    Output,
    GetResult,
    get_custom,
    SchedulingGroupBarrier,
    MMA,
    NewRegister,
)
from ..utils.general_utils import get_induction_variable
from ..utils.graph_utils import replace_uses_in
import torch.fx as fx
from collections import deque, defaultdict
from ..visualization import visualize_mapped_graphs, visualize_graph
from ....support.logging import get_logger
from .loop_reconstruction_utils import (
    ArgumentContext,
    create_fill_stage_schedule,
    create_drain_stage_schedule,
    liveness_analysis,
    partition_graph_by_stage,
    interleave_instructions,
)
from .resources import get_custom_operation_type
from enum import Enum

logger = get_logger("turbine.wave.scheduling.loop_reconstruction")


class PipelineStage(Enum):
    PROLOGUE = 0
    KERNEL = 1
    EPILOGUE = 2


def add_nodes_by_schedule(
    reduction_graph: fx.Graph,
    partitioned_graph: list[dict[int, fx.Node]],
    arg_context: ArgumentContext,
    stages: list[int],
    initiation_interval: int,
    induction_variable: IndexSymbol,
    current_induction_variables: list[int],
    rotating_registers: dict[fx.Node, list[fx.Node]],
    pipelining_stage: PipelineStage = PipelineStage.KERNEL,
    use_scheduling_barriers: bool = False,
):
    """
    Interleave the instructions in the partitioned graph by stage
    for a single initiation interval, updating the argument maps
    per stage starting at the provided start times and indices.
    """
    fill_or_drain = pipelining_stage in [PipelineStage.PROLOGUE, PipelineStage.EPILOGUE]

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

        instructions = defaultdict(int)
        for iteration, stage, node in interleaved_instructions:
            logger.debug(f"Node: {node}, Stage: {stage}, Iteration: {iteration}")
            custom_node = get_custom(node)
            logger.debug(f"Node args: {node.args}")
            preferred_stage = (
                stage if pipelining_stage == PipelineStage.KERNEL else None
            )
            for arg in node.args:
                if arg_context.contains_in_iteration(iteration, arg):
                    logger.debug(
                        f"Found arg: {arg} at iteration {iteration} in partitioned argument map. Using {arg_context.get_from_iteration(iteration, arg, preferred_stage)}."
                    )
                    continue
            new_node = custom_node.copy(
                new_name=node.name,
                new_graph=reduction_graph,
                arg_transform=lambda x: (
                    arg_context.get_from_iteration(iteration, x, preferred_stage)
                    if arg_context.contains_in_iteration(iteration, x)
                    else x
                ),
            )
            instructions[get_custom_operation_type(new_node)] += 1
            # Update the argument context.
            arg_context[(iteration, stage, node)] = new_node.fx_node
            logger.debug(
                f"Copying Node: {node}, Stage: {stage}, Iteration: {iteration} -> {new_node.fx_node}"
            )
            # Set the index for the new node by substituting the induction variable
            # for the current iteration.
            new_node.index = node.index
            if new_node.index:
                for dim in new_node.index:
                    new_node.index[dim] = new_node.index[dim].subs(
                        {induction_variable: current_induction_variables[iteration]}
                    )
            if custom_node.expanded_dims:
                new_node.expanded_dims = custom_node.expanded_dims
            # Add scheduling parameters for debugging.
            new_node.scheduling_parameters = node.scheduling_parameters
            # Update the rotating registers and argument context for the current node (if applicable).
            old_node = None
            if node in rotating_registers:
                rotating_registers[node].append(new_node.fx_node)
                old_node = rotating_registers[node].popleft()
                # If draining, then override the rotating registers and update the argument context.
                if fill_or_drain:
                    for next_stage in range(stage + 1, len(stages)):
                        arg_context[(iteration, next_stage, node)] = new_node.fx_node

            # Update the iter and init args in the argument context whenever a result is computed.
            if node in arg_context.results:
                iter_arg = arg_context.result_to_iter_arg[node]
                logger.debug(
                    f"Updating result: {node} -> {iter_arg} to {new_node.fx_node}."
                )
                arg_context.map_arg_all_after_iteration(
                    iter_arg,
                    new_node.fx_node,
                    iteration,
                )
                # In situations where we have an iter_arg as a rotating register,
                # we also have the output as a rotating register. So when we
                # are updating the output, we update the iter_arg as well with the
                # old value of the output rotating register. Consider this example:
                # Say we have the following:
                #
                # Stage 0:
                # iter_arg0
                #
                #
                # output = compute(...) -> here we update iter_arg0 to have the output value
                #                          for the next stage, so that it gets picked up in stage1.
                #
                # Stage 1:
                # b = use(iter_arg0)
                if (
                    pipelining_stage == PipelineStage.KERNEL
                    or pipelining_stage == PipelineStage.PROLOGUE
                ):
                    if iter_arg in rotating_registers and old_node:
                        logger.debug(
                            f"Updating rotating register iter arg {iter_arg} -> {old_node}."
                        )
                        rotating_registers[iter_arg].append(old_node)
                        rotating_registers[iter_arg].popleft()
                        for next_stage in range(stage + 1, len(stages)):
                            arg_context[(iteration, next_stage, iter_arg)] = old_node
                if pipelining_stage == PipelineStage.PROLOGUE:
                    logger.debug(
                        f"Updating result: {node} -> {arg_context.result_to_init_arg[node]} to {new_node.fx_node}."
                    )
                    arg_context.map_arg_all_after_iteration(
                        arg_context.result_to_init_arg[node],
                        new_node.fx_node,
                        iteration,
                    )

        if pipelining_stage == PipelineStage.KERNEL and use_scheduling_barriers:
            SchedulingGroupBarrier(instructions, 0).add_to_graph(reduction_graph)


def push_placeholders(
    implicit_captures: list[fx.Node],
    reduction_subgraph: fx.Node,
    arg_context: ArgumentContext,
):
    """
    Push placeholders into the argument context for the reduction graph.
    """
    for node in reduction_subgraph.nodes:
        custom = get_custom(node)
        if isinstance(custom, Placeholder) and not isinstance(custom, IterArg):
            root_node = [x for x in implicit_captures if x.name == node.name][0]
            assert root_node is not None
            arg_context.map_arg_all(node, root_node)


def add_missing_registers(graph: fx.Graph):
    """
    This function goes through the graph and finds MMA operators whose accumulator
    is undefined (not in the current graph). For those operators, it replaces the accumulator
    with a new register of the same shape and type and with the same index.

    This is necessary in situations where the register is defined inside the loop
    but when we are trying to insert MMA operators outside the loop (such as for the
    prologue and epilogue).
    """
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, MMA):
            acc = get_custom(custom.acc)
            if acc.graph != custom.graph:
                with custom.graph.inserting_before(node):
                    register = NewRegister(
                        acc.shape, acc.dtype, acc.value
                    ).add_to_graph(custom.graph)
                    register.index = acc.index
                    custom.update_arg("acc", register)


def construct_prologue(
    reduction_subgraph: fx.Graph,
    reduction: Iterate,
    partitioned_graph: list[dict[int, fx.Node]],
    num_stages: int,
    initiation_interval: int,
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
    logger.debug("=====================================")
    logger.debug("Constructing prologue.")
    logger.debug("=====================================")

    arg_context = ArgumentContext(
        reduction.outputs(reduction_subgraph),
        reduction.iter_args(reduction_subgraph),
        reduction.init_args,
        num_stages,
    )

    # Map iter args to init args in the prologue.
    original_init_args = list(reduction.init_args)
    for iter_arg, init_arg in zip(
        reduction.iter_args(reduction_subgraph), reduction.init_args
    ):
        arg_context.map_arg_all(iter_arg, init_arg)

    push_placeholders(reduction.implicit_captures, reduction_subgraph, arg_context)
    with reduction.graph.inserting_before(reduction.fx_node):
        for i in range(num_stages - 1):
            add_nodes_by_schedule(
                reduction.graph,
                partitioned_graph,
                arg_context,
                stages[i],
                initiation_interval,
                induction_variable,
                new_induction_variables,
                rotating_registers,
                PipelineStage.PROLOGUE,
            )

    # During the prologue, we may have computed results that need to be passed as init args
    # to the kernel.
    new_init_args: list[fx.Node] = []
    for init_arg in reduction.init_args:
        mapped_init_arg = arg_context.lookup(init_arg)
        if mapped_init_arg is None:
            mapped_init_arg = init_arg
        logger.debug(f"Mapping init_arg {init_arg} -> {mapped_init_arg}.")
        new_init_args.append(mapped_init_arg)
    reduction.init_args = new_init_args

    # We may also have some iter_args as rotating registers. These will need
    # to be initialized to the original init args which we do here.
    iter_args = reduction.iter_args(reduction_subgraph)
    for node, registers in rotating_registers.items():
        if node in iter_args:
            if all(x is None for x in registers) and len(registers) == 1:
                registers[0] = original_init_args[iter_args.index(node)]

    # Add missing registers. Since registers are not present
    # in the scheduling code, we could end up with a situation where
    # we move mma ops outside the reduction that do not have a corresponding
    # register. We remedy this in the function below.
    add_missing_registers(reduction.graph)


def flatten_dict_values(
    rotating_registers: dict[fx.Node, list[fx.Node]],
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
    assert count == sum(rotating_registers_shapes.values())
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
    we evaluate which stage it belongs to and update the argument
    context for the next stage and n - 1 stages after it, where
    n is the total number of rotating registers.
    If var a has [a, b, c] as rotating registers, then in a 3-stage schedule
        a is used in stage 2, (iteration 0)
        b in stage 1, (iteration 1)
        c in stage 0. (iteration 2)
    """
    new_rotating_registers: dict[fx.Node, deque[fx.Node]] = {}
    count = 0
    iter_arg_count = len(arg_context.iter_args)
    for node, registers in rotating_registers.items():
        new_registers: deque[fx.Node] = deque()
        custom = get_custom(node)
        stage = custom.scheduling_parameters["stage"]
        iteration = arg_context.get_kernel_iteration(stage)
        if node not in arg_context.iter_args:
            arg_context[(iteration, stage, node)] = registers[-1]
        for i, register in enumerate(registers):
            if create_new_nodes:
                mapped_stage = stage + len(registers) - i
                mapped_iteration = arg_context.get_kernel_iteration(mapped_stage)
                iter_arg = IterArg(f"rotating_reg_{count}").add_to_graph(graph)
                iter_arg.type = get_custom(node).type
                iter_arg.index = get_custom(node).index
                iter_arg.iter_idx = iter_arg_count
                iter_arg_count += 1
                new_registers.append(iter_arg)
                mapped_value = iter_arg
            else:
                mapped_stage = stage + len(registers) - i - 1
                mapped_iteration = arg_context.get_kernel_iteration(mapped_stage)
                mapped_value = register
            arg_context[(mapped_iteration, mapped_stage, node)] = mapped_value
            logger.debug(
                f"Mapped orig: {node_map[node]} / mapped: {mapped_value} to stage {mapped_stage} at iteration {mapped_iteration}."
            )
            count += 1
        if new_registers:
            new_rotating_registers[node] = new_registers
    return new_rotating_registers


def construct_kernel(
    reduction_subgraph: fx.Graph,
    reduction: Iterate,
    partitioned_graph: list[dict[int, fx.Node]],
    num_stages: int,
    initiation_interval: int,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
    node_map: dict[fx.Node, fx.Node],
    visualize: bool = False,
    use_scheduling_barriers: bool = False,
) -> tuple[Iterate, fx.Graph]:
    """
    Construct the kernel of the pipelined loop.
    First, we construct a new reduction op with an empty graph.
    Then, we set the init args, construct the iter args and add the ops.
    Finally, we create the output node with the return values.
    The iter args/results of the pipelined reduction are always:
    [results0, result1, ..., resultN, rotating_reg0, rotating_reg1, ..., rotating_regN]
    """
    logger.debug("=====================================")
    logger.debug("Constructing kernel.")
    logger.debug("=====================================")

    with reduction.graph.inserting_before(reduction.fx_node):
        pipelined_reduction = Iterate(
            reduction.axis,
            init_args=reduction.init_args + flatten_dict_values(rotating_registers),
            subgraph_name="pipelined_iterate",
            implicit_captures=reduction.implicit_captures,
        ).add_to_graph(reduction.graph, type=reduction.type)
        pipelined_reduction.index = reduction.index
        pipelined_reduction_graph = fx.Graph()
        reduction.graph.subgraphs["pipelined_iterate"] = pipelined_reduction_graph

        # Update the argument map for the new reduction.
        arg_context = ArgumentContext(
            reduction.outputs(reduction_subgraph),
            reduction.iter_args(reduction_subgraph),
            reduction.init_args,
            num_stages,
        )
        push_placeholders(reduction.implicit_captures, reduction_subgraph, arg_context)

        # For the original iter args, we just map the old ones to the new ones.
        # Do this for all stages, since the original iter args are "dummy" nodes
        # during scheduling.
        for node in arg_context.iter_args:
            iter_arg = IterArg(node.name).add_to_graph(pipelined_reduction_graph)
            iter_arg.type = get_custom(node).type
            iter_arg.index = get_custom(node).index
            iter_arg.iter_idx = get_custom(node).iter_idx
            arg_context.map_arg_all(node, iter_arg)

        # Push the rotating registers into the argument context.
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
            list(reversed(range(num_stages))),
            initiation_interval,
            induction_variable,
            new_induction_variables,
            new_rotating_registers,
            PipelineStage.KERNEL,
            use_scheduling_barriers,
        )

        # Create output node (last node in the graph).
        return_vals: list[fx.Node] = arg_context.get_kernel_results()
        for registers in new_rotating_registers.values():
            return_vals.extend(registers)

        Output(return_vals).add_to_graph(pipelined_reduction_graph)
        reduction.replace_all_uses_with(pipelined_reduction)

        if visualize:
            visualize_mapped_graphs(
                pipelined_reduction_graph,
                new_rotating_registers,
                arg_context.argument_map,
                "kernel.png",
            )

        # Add missing registers. Since registers are not present
        # in the scheduling code, we could end up with a situation where
        # we move mma ops outside the reduction that do not have a corresponding
        # register. We remedy this in the function below.
        add_missing_registers(pipelined_reduction_graph)

        return pipelined_reduction, pipelined_reduction_graph


def construct_epilogue(
    reduction_subgraph: fx.Graph,
    reduction: Iterate,
    pipelined_reduction: Iterate,
    partitioned_graph: list[dict[int, fx.Node]],
    num_stages: int,
    initiation_interval: int,
    rotating_registers: dict[fx.Node, list[fx.Node]],
    induction_variable: IndexSymbol,
    new_induction_variables: list[int],
    stages: list[int],
    num_rotating_registers: dict[fx.Node, int],
    node_map: dict[fx.Node, fx.Node],
    visualize: bool = False,
):
    """
    Construct the epilogue of the pipelined loop.
    The difference from the prologue is that we need to map the results
    of the pipelined reduction to the remaining stages. (In the prologue,
    no iteration is every completed and so we don't compute the final results)
    We emit GetResult nodes for the rotating registers and map them to
    the different epilogue stages.
    """
    logger.debug("=====================================")
    logger.debug("Constructing epilogue.")
    logger.debug("=====================================")

    arg_context = ArgumentContext(
        reduction.outputs(reduction_subgraph),
        reduction.iter_args(reduction_subgraph),
        reduction.init_args,
        num_stages,
    )

    existing_get_results: list[GetResult] = [
        x for x in pipelined_reduction.users if isinstance(x, GetResult)
    ]
    existing_indices = [x.res_idx for x in existing_get_results]

    # Map the results from the kernel to the init args (for stages).
    # The number of iter args may not be the same as the number of get results
    # and so we have to add additional get results for the missing iter args.
    # This happens if some of the iter args have no uses outside the reduction
    # (such as the max value in flash attention). While they may not have any
    # uses in the original reduction, they will have uses in the pipelined
    # reduction outside the reduction and so need to be added in the correct order.
    iter_args = reduction.iter_args(reduction_subgraph)
    last_get_result = existing_get_results[-1].fx_node
    for i in range(len(iter_args)):
        if i in existing_indices:
            continue
        with pipelined_reduction.graph.inserting_before(
            existing_get_results[-1].fx_node.next
        ):
            result = GetResult(pipelined_reduction.fx_node, i).add_to_graph(
                pipelined_reduction.graph, type=iter_args[i].type
            )
            existing_get_results.append(get_custom(result))
            last_get_result = result

    existing_get_results = sorted(existing_get_results, key=lambda x: x.res_idx)
    existing_users = {x: x.users for x in existing_get_results}

    for iter_arg, get_result in zip(iter_args, existing_get_results):
        arg_context.map_arg_all(iter_arg, get_result.fx_node)

    with pipelined_reduction.graph.inserting_before(last_get_result.next):
        # Add get result nodes for the rotating registers and update the
        # argument map with them.
        rotating_registers_get_results = []
        offset = len(existing_get_results)
        flattened_rotating_registers = flatten_dict_values(rotating_registers)
        for i in range(len(flattened_rotating_registers)):
            rotating_registers_get_results.append(
                GetResult(pipelined_reduction.fx_node, i + offset).add_to_graph(
                    pipelined_reduction.graph,
                    type=flattened_rotating_registers[i].type,
                )
            )
        rotating_registers = unflatten_dict_values(
            num_rotating_registers, rotating_registers_get_results
        )

        # Push the rotating registers onto the argument map.
        push_rotating_registers(arg_context, rotating_registers, None, node_map, False)
        push_placeholders(reduction.implicit_captures, reduction_subgraph, arg_context)

        for i in range(num_stages - 1):
            add_nodes_by_schedule(
                pipelined_reduction.graph,
                partitioned_graph,
                arg_context,
                stages[i],
                initiation_interval,
                induction_variable,
                new_induction_variables,
                rotating_registers,
                PipelineStage.EPILOGUE,
            )

        # Replace the existing uses with the new results.
        new_results = arg_context.get_mapped_results(existing_get_results)
        assert len(new_results) == len(existing_get_results)
        for i, get_result in enumerate(existing_get_results):
            replace_uses_in(existing_users, get_result, new_results[i])

        # Add missing registers. Since registers are not present
        # in the scheduling code, we could end up with a situation where
        # we move mma ops outside the reduction that do not have a corresponding
        # register. We remedy this in the function below.
        add_missing_registers(pipelined_reduction.graph)

        if visualize:
            visualize_mapped_graphs(
                pipelined_reduction.graph,
                rotating_registers,
                arg_context.argument_map,
                "epilogue.png",
            )


def construct_pipelined_loop(
    trace: CapturedTrace,
    reduction: Iterate,
    graph: fx.Graph,
    constraints: list[Constraint],
    num_stages: int,
    initiation_interval: int,
    node_map: dict[fx.Node, fx.Node],
    max_induction_variable: int,
    visualize: bool = False,
    use_scheduling_barriers: bool = False,
) -> fx.Node:
    """
    Given a graph annotated with scheduling parameters, construct a pipelined loop
    with a prologue, kernel and epilogue.
    """
    induction_variable = get_induction_variable(reduction, constraints)
    num_rotating_registers = liveness_analysis(graph, reduction)
    rotating_registers: dict[fx.Node, deque[fx.Node]] = {
        k: deque([None for _ in range(v)]) for k, v in num_rotating_registers.items()
    }
    partitioned_graph = partition_graph_by_stage(graph, num_stages)
    # Construct prologue.
    construct_prologue(
        graph,
        reduction,
        partitioned_graph,
        num_stages,
        initiation_interval,
        rotating_registers,
        induction_variable,
        list(range(num_stages)),
        create_fill_stage_schedule(num_stages),
    )
    # Construct kernel.
    pipelined_reduction, pipelined_reduction_graph = construct_kernel(
        graph,
        reduction,
        partitioned_graph,
        num_stages,
        initiation_interval,
        rotating_registers,
        induction_variable,
        [induction_variable + i for i in range(num_stages)],
        node_map,
        visualize,
        use_scheduling_barriers,
    )
    trace.add_subgraph(
        get_custom(pipelined_reduction).subgraph_name, pipelined_reduction_graph
    )
    # Construct epilogue.
    construct_epilogue(
        graph,
        reduction,
        get_custom(pipelined_reduction),
        partitioned_graph,
        num_stages,
        initiation_interval,
        rotating_registers,
        induction_variable,
        [max_induction_variable - num_stages + i for i in range(num_stages)],
        create_drain_stage_schedule(num_stages),
        num_rotating_registers,
        node_map,
        visualize,
    )

    # Remove the unpipelined reduction.
    reduction.graph.erase_node(reduction.fx_node)

    if visualize:
        visualize_graph(pipelined_reduction.graph, "pipelined.png")

    return pipelined_reduction
