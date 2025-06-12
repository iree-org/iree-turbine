# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from iree.turbine.kernel.ops.wave_ops import (
    NestedRegionOp,
    Placeholder,
    get_custom,
    Allocate,
    Read,
    Write,
)
from ..._support.tracing import CapturedTrace
from typing import Sequence, Optional, Dict, Tuple, Callable
from ..scheduling.graph_utils import Edge, EdgeWeight
import numpy as np
import timeit
import logging

logger = logging.getLogger(__name__)


def get_node_type(node: fx.Node) -> str:
    """Helper function to determine the type of operation a node represents."""
    custom = get_custom(node)
    if custom is None:
        return "Unknown"

    # Special handling for Read/Write operations.
    # Note: The reason we cannot use custom.memory_type.address_space is because we are printing
    # a "scheduling copy" of the original fx.graph.  This is not the original fx.graph but only
    # contains nodes that we are interested in scheduling.
    #
    # As a result, this graph contains no placeholders since we don't want to schedule them
    # and so we can't use custom.memory_type.address_space because the global reads/writes have
    # no placeholders and custom.memory is None.
    if isinstance(custom, (Read, Write)):
        base_type = "Read" if isinstance(custom, Read) else "Write"
        uses_allocate = custom.memory and isinstance(
            get_custom(custom.memory), Allocate
        )
        memory_type = "Global" if not uses_allocate else "Shared"
        return f"{base_type}{memory_type}"

    # Fallback to class name if type is not recognized
    return custom.__class__.__name__


def _write_metadata_to_file(f, initiation_interval: int, num_stages: int):
    """Write schedule metadata to a file."""
    f.write(f"Initiation Interval: {initiation_interval}\n")
    f.write(f"Number of Stages: {num_stages}\n")


def _write_rrt_to_file(
    f,
    resource_reservations: np.ndarray,
    resource_names: list[str],
    initiation_interval: int,
):
    """Write resource reservation table to a file."""
    if resource_reservations is None or resource_names is None:
        return

    f.write("\n# Resource Reservation Table (RRT):\n")
    f.write("# Each row represents a cycle in the initiation interval\n")
    f.write("# Each column represents a resource type\n")
    f.write("# Format: cycle | resource_usage\n")
    f.write("\n")

    # Calculate column widths for RRT
    rrt_col_widths = {
        "cycle": max(len("Cycle"), len(str(initiation_interval - 1))),
        "resources": [
            max(len(name), 3) for name in resource_names
        ],  # At least 3 chars for numbers
    }

    # Create a header line with resource names
    header_parts = [f"{'Cycle':>{rrt_col_widths['cycle']}}"]
    for name, width in zip(resource_names, rrt_col_widths["resources"]):
        header_parts.append(f"{name:>{width}}")
    f.write(" | ".join(header_parts) + "\n")

    # Write a separator line
    separator_parts = ["-" * rrt_col_widths["cycle"]]
    separator_parts.extend("-" * width for width in rrt_col_widths["resources"])
    f.write(" | ".join(separator_parts) + "\n")

    # Write the resource values
    for cycle in range(initiation_interval):
        # Format each resource value with its column width
        resource_str = " | ".join(
            f"{int(r):>{width}d}"
            for r, width in zip(
                resource_reservations[cycle], rrt_col_widths["resources"]
            )
        )
        f.write(f"{cycle:>{rrt_col_widths['cycle']}d} | {resource_str}\n")
    f.write("\n")


def _parse_rrt_from_lines(
    lines: list[str], initiation_interval: int
) -> Tuple[Optional[np.ndarray], Optional[list[str]], Optional[int]]:
    """Parse resource reservation table from file lines.
    Returns a tuple of (resource_reservations, resource_names, rrt_end_line) or (None, None, None) if no RRT found.
    """
    rrt_start_line = -1
    rrt_end_line = -1

    # Find RRT section boundaries
    comment_and_header_offset = 4 + 2
    for i, line in enumerate(lines):
        if line.startswith("# Resource Reservation Table (RRT):"):
            rrt_start_line = i
            # 4 lines for the comments, 2 lines for the header and separator, and then the RRT data
            rrt_end_line = (
                rrt_start_line + comment_and_header_offset + initiation_interval
            )

    if rrt_start_line == -1 or rrt_end_line == -1:
        return None, None, None

    rrt_lines = lines[rrt_start_line : rrt_end_line + 1]
    # Find the header line with resource names
    header_line = None
    for line in rrt_lines:
        if not line.startswith("#") and "Cycle" in line and "|" in line:
            header_line = line
            break

    if not header_line:
        return None, None, None

    # Extract resource names from header
    resource_names = [name.strip() for name in header_line.split("|")[1:]]
    num_resources = len(resource_names)

    # Initialize RRT array
    resource_reservations = np.zeros(
        (initiation_interval, num_resources), dtype=np.int32
    )

    # Parse RRT data lines
    for line in rrt_lines[comment_and_header_offset + 1 :]:
        # Parse cycle and resource values
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == num_resources + 1:  # +1 for cycle column
            cycle = int(parts[0].strip())
            values = [int(v.strip()) for v in parts[1:]]
            resource_reservations[cycle] = values

    return resource_reservations, resource_names, rrt_end_line


def _parse_metadata_from_lines(lines: list[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse schedule metadata from file lines.
    Returns a tuple of (initiation_interval, num_stages) or (None, None) if metadata not found.
    """
    initiation_interval = None
    num_stages = None

    for line in lines:
        if line.startswith("Initiation Interval:"):
            initiation_interval = int(line.split(":")[1].strip())
        elif line.startswith("Number of Stages:"):
            num_stages = int(line.split(":")[1].strip())

    return initiation_interval, num_stages


def _calculate_column_widths(
    schedule_data: list[Tuple[fx.Node, int, int, str, list[str], int]],
) -> dict[str, int]:
    """Calculate column widths for schedule table based on data."""
    col_widths = {
        "name": len("Node Name"),
        "type": len("Node Type"),
        "sort_key": len("Node Sort Key"),
        "cycle": len("Cycle"),
        "relative_cycle": len("Relative Cycle"),
        "stage": len("Stage"),
        "users": len("User Sort Keys"),
    }

    for (
        node,
        cycle,
        stage,
        node_type,
        user_sort_keys,
        initiation_interval,
    ) in schedule_data:
        col_widths["name"] = max(col_widths["name"], len(str(node.name)))
        col_widths["type"] = max(col_widths["type"], len(node_type))
        col_widths["sort_key"] = max(col_widths["sort_key"], len(str(node._sort_key)))
        col_widths["cycle"] = max(col_widths["cycle"], len(str(cycle)))
        col_widths["relative_cycle"] = max(
            col_widths["relative_cycle"], len(str(cycle % initiation_interval))
        )
        col_widths["stage"] = max(col_widths["stage"], len(str(stage)))
        col_widths["users"] = max(col_widths["users"], len(", ".join(user_sort_keys)))

    # Add padding
    for key in col_widths:
        col_widths[key] += 2  # Add 2 spaces padding on each side

    return col_widths


def _write_schedule_header(f, col_widths: dict[str, int]):
    """Write schedule table header to file."""
    header = (
        f"{'Node Name':<{col_widths['name']}} | "
        f"{'Node Type':<{col_widths['type']}} | "
        f"{'Node Sort Key':<{col_widths['sort_key']}} | "
        f"{'Cycle':<{col_widths['cycle']}} | "
        f"{'Relative Cycle':<{col_widths['relative_cycle']}} | "
        f"{'Stage':<{col_widths['stage']}} | "
        f"{'User Sort Keys':<{col_widths['users']}}"
    )
    f.write(header + "\n")

    # Write separator
    separator = (
        f"{'-' * col_widths['name']} | "
        f"{'-' * col_widths['type']} | "
        f"{'-' * col_widths['sort_key']} | "
        f"{'-' * col_widths['cycle']} | "
        f"{'-' * col_widths['relative_cycle']} | "
        f"{'-' * col_widths['stage']} | "
        f"{'-' * col_widths['users']}"
    )
    f.write(separator + "\n")


def _write_schedule_data(
    f,
    schedule_data: list[Tuple[fx.Node, int, int, str, list[str], int]],
    col_widths: dict[str, int],
):
    """Write schedule data to file."""
    separator = (
        f"{'-' * col_widths['name']} | "
        f"{'-' * col_widths['type']} | "
        f"{'-' * col_widths['sort_key']} | "
        f"{'-' * col_widths['cycle']} | "
        f"{'-' * col_widths['relative_cycle']} | "
        f"{'-' * col_widths['stage']} | "
        f"{'-' * col_widths['users']}"
    )

    current_stage = None
    for (
        node,
        cycle,
        stage,
        node_type,
        user_sort_keys,
        initiation_interval,
    ) in schedule_data:
        # Add stage separator if we're moving to a new stage
        if current_stage is not None and stage != current_stage:
            f.write(separator + "\n")
        current_stage = stage

        row = (
            f"{str(node.name):<{col_widths['name']}} | "
            f"{node_type:<{col_widths['type']}} | "
            f"{str(node._sort_key):<{col_widths['sort_key']}} | "
            f"{str(cycle):<{col_widths['cycle']}} | "
            f"{str(cycle % initiation_interval):<{col_widths['relative_cycle']}} | "
            f"{str(stage):<{col_widths['stage']}} | "
            f"{', '.join(user_sort_keys):<{col_widths['users']}}"
        )
        f.write(row + "\n")


def dump_schedule(
    graph: fx.Graph,
    schedule: Dict[fx.Node, int],
    initiation_interval: int,
    num_stages: int,
    dump_file: str,
    resource_reservations: Optional[np.ndarray] = None,
    resource_names: Optional[list[str]] = None,
):
    """
    Dumps the schedule to a file in pipe-delimited table format.
    Each row contains: node_name | node_type | node_sort_key | cycle | relative_cycle | stage | user_sort_keys
    The schedule metadata (II and num_stages) is stored at the top.
    If provided, the resource reservation table is also stored.
    Columns are padded to ensure pipe alignment.
    Rows are sorted by cycle first, then by stage.
    A separator line of dashes is added between different stages.
    """
    # Prepare schedule data for sorting
    schedule_data = []
    for node, cycle in schedule.items():
        user_sort_keys = [str(u._sort_key) for u in node.users]
        node_type = get_node_type(node)
        stage = cycle // initiation_interval
        schedule_data.append(
            (node, cycle, stage, node_type, user_sort_keys, initiation_interval)
        )

    # Sort the schedule data by cycle first, then by stage
    schedule_data.sort(key=lambda x: (x[1], x[2]))  # Sort by (cycle, stage)

    # Calculate column widths
    col_widths = _calculate_column_widths(schedule_data)

    with open(dump_file, "w") as f:
        # Write metadata
        _write_metadata_to_file(f, initiation_interval, num_stages)

        # Write RRT if provided
        _write_rrt_to_file(
            f, resource_reservations, resource_names, initiation_interval
        )

        # Write schedule table
        _write_schedule_header(f, col_widths)
        _write_schedule_data(f, schedule_data, col_widths)


def _parse_sort_key(sort_key_str: str) -> tuple:
    """Safely parse a sort key string representation into a tuple of integers.
    The input format is expected to be like "(1, 2, 3)" or "(1,)".
    Handles whitespace and empty elements.
    """
    # Remove parentheses and split by commas
    inner = sort_key_str.strip("()")
    if not inner:  # Handle empty tuple case
        return tuple()
    # Split by comma, filter out empty strings, and convert each part to integer
    return tuple(int(x.strip()) for x in inner.split(",") if x.strip())


def load_schedule(
    load_file: str, graph: fx.Graph
) -> Tuple[
    Dict[fx.Node, int],
    int,
    int,
    list[fx.Node],
    list[Edge],
    Optional[np.ndarray],
    Optional[list[str]],
]:
    """
    Loads a schedule from a file in pipe-delimited table format.
    Returns:
        - schedule: Dictionary mapping nodes to their cycles
        - initiation_interval: The initiation interval
        - num_stages: Number of stages
        - nodes: List of all nodes in the schedule
        - edges: List of Edge objects representing dependencies
        - resource_reservations: Optional numpy array of shape (II, num_resources) containing resource reservations
        - resource_names: Optional list of resource names corresponding to each column in resource_reservations
    The stage column is ignored as it can be calculated from cycle and II.
    The relative_cycle column is ignored as it can be calculated from cycle and II.
    Separator lines (lines containing only dashes and pipes) are skipped.
    """
    with open(load_file, "r") as f:
        content = f.read()

    # Parse file content
    lines = content.strip().split("\n")

    # Get metadata
    initiation_interval, num_stages = _parse_metadata_from_lines(lines)
    if initiation_interval is None or num_stages is None:
        raise ValueError("Schedule file missing required metadata (II or num_stages)")

    # Get RRT data if present
    resource_reservations, resource_names, rrt_end_line = _parse_rrt_from_lines(
        lines, initiation_interval
    )

    assert rrt_end_line is not None, "RRT not found in schedule file"

    # Parse schedule data
    data_lines = [
        l
        for l in lines[rrt_end_line + 1 :]
        if l and not l.startswith("#") and "|" in l and not all(c in "-| " for c in l)
    ][
        1:
    ]  # Skip the header line

    # Map node names back to node objects
    node_map = {node._sort_key: node for node in graph.nodes}
    schedule = {}
    edges = []
    nodes = set()  # Use a set to avoid duplicates

    for line in data_lines:
        # Parse all columns (now including relative_cycle)
        name, _, sort_key_str, cycle_str, _, _, user_sort_keys_str = [
            x.strip() for x in line.split("|")
        ]
        # Convert sort key string to tuple safely
        sort_key = _parse_sort_key(sort_key_str)
        if sort_key in node_map:
            node = node_map[sort_key]
            schedule[node] = int(cycle_str)
            nodes.add(node)

            # Parse user sort keys and create edges
            if user_sort_keys_str:
                for user_key_str in user_sort_keys_str.split(","):
                    user_key = _parse_sort_key(user_key_str.strip())
                    if user_key in node_map:
                        user_node = node_map[user_key]
                        # Create Edge with default EdgeWeight
                        edge = Edge(node, user_node, EdgeWeight(0, 0))
                        edges.append(edge)
                        nodes.add(user_node)

    # Convert nodes set to list for consistent return type
    nodes_list = list(nodes)

    return (
        schedule,
        initiation_interval,
        num_stages,
        nodes_list,
        edges,
        resource_reservations,
        resource_names,
    )


def print_graph(graph: fx.Graph, printer: Callable = print):
    """
    Pretty-print the graph containing this node.
    """
    graph_str = str(graph)
    graph_str = graph_str.replace(
        "iree.turbine.kernel.lang.kernel_buffer.KernelBufferMeta.new_subtype.<locals>.SubType",
        "",
    )
    graph_str = graph_str.replace("target=iree.turbine.kernel.ops.wave_ops.", "")
    graph_str = graph_str.replace("call_function", "")
    printer(graph_str)


def print_trace(
    trace: CapturedTrace, custom_print: bool = True, printer: Callable = print
):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for name, subgraph in reversed(list(trace.region_graph.subgraphs.items())):
        if name == trace.root_graph:
            name = f"{name} [root]"
        printer(f"{name}:\n")
        print_graph(subgraph, printer)
        if custom_print:
            printer("Custom format:")
            for node in subgraph.nodes:
                printer(get_custom(node))


def print_subgraph(
    trace: CapturedTrace,
    subgraph_name: str,
    custom_print: bool = True,
    printer: Callable = print,
):
    """
    Prints a specific subgraphs of a trace.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    for name, subgraph in trace.region_graph.subgraphs.items():
        if name == subgraph_name:
            printer(subgraph)
            if custom_print:
                for node in subgraph.nodes:
                    printer(get_custom(node))


def try_apply_pass(
    p,
    trace: CapturedTrace,
    print_ir_before: Sequence[str] = [],
    print_ir_after: Sequence[str] = [],
    pass_times: Optional[dict[str, float]] = None,
):
    pass_name = p.__name__
    if "all" in print_ir_before or pass_name in print_ir_before:
        print(f"***Before {pass_name}***\n")
        print_trace(trace)
    try:
        start = timeit.default_timer()
        p()
        end = timeit.default_timer()
        if pass_times is not None:
            print_name = pass_name
            if pass_name in pass_times:
                # Make name unique by adding a counter if it already exists
                counter = 1
                while f"{pass_name}_{counter}" in pass_times:
                    counter += 1
                print_name = f"{pass_name}_{counter}"

            pass_times[print_name] = end - start
    except Exception:
        logger.info(f"Error in pass: {pass_name}\n")
        print_trace(trace, logger.info)
        raise
    if "all" in print_ir_after or pass_name in print_ir_after:
        print(f"***After {pass_name}***\n")
        print_trace(trace)


def print_mlir_style(trace: CapturedTrace, name: str):
    """
    Prints a graph in MLIR style format.
    """
    # First pass: collect node names from the parent graph and all subgraphs
    node_names = {}
    graph = trace.get_root_graph()

    def collect_node_names(g):
        for node in g.nodes:
            custom = get_custom(node)
            if isinstance(custom, Placeholder):
                node_names[node] = custom._name
            else:
                node_names[node] = f"{node.name}"
            if isinstance(custom, NestedRegionOp):
                subgraph = g.subgraphs[custom.subgraph_name]
                collect_node_names(subgraph)

    collect_node_names(graph)

    # Second pass: print operations
    print(f"func.func @{name}(", end="")
    # Print function arguments
    args = []
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, Placeholder):
            args.append(f"%{node_names[node]}: {_format_mlir_type(custom.type)}")
    print(", ".join(args), end="")
    print(") {")

    # Print operations
    for node in graph.nodes:
        custom = get_custom(node)
        if isinstance(custom, Placeholder):
            continue
        elif isinstance(custom, NestedRegionOp):
            # Print iterate operation
            implicit_captures_str = (
                "["
                + ", ".join(
                    f"%{node_names[capture]}" for capture in custom.implicit_captures
                )
                + "]"
            )
            print(
                f'  %{node_names[node]} = wave.iterate({custom.axis}, {custom.init_args}, "{custom.subgraph_name}", {implicit_captures_str}, {custom.start}, {custom.condition}) : {_format_mlir_type(custom.type)}'
            )
            # Print subgraph contents
            print("  {")
            subgraph = graph.subgraphs[custom.subgraph_name]
            # Create a mapping from placeholder nodes to their corresponding captured nodes
            placeholder_to_capture = {}
            for placeholder_node in subgraph.nodes:
                if isinstance(get_custom(placeholder_node), Placeholder):
                    for i, capture in enumerate(custom.implicit_captures):
                        if placeholder_node.name == capture.name:
                            placeholder_to_capture[placeholder_node] = capture
                            break

            for subnode in subgraph.nodes:
                subcustom = get_custom(subnode)
                if isinstance(subcustom, Placeholder):
                    continue
                print(
                    f"    %{subnode.name} = {_convert_to_mlir_op(subcustom.tkw_op_name)}(",
                    end="",
                )
                args = []
                for arg in subnode.args:
                    if isinstance(arg, fx.Node):
                        if arg in placeholder_to_capture:
                            args.append(f"%{node_names[placeholder_to_capture[arg]]}")
                        else:
                            args.append(f"%{node_names[arg]}")
                    elif isinstance(arg, fx.Proxy):
                        args.append(f"%{node_names[arg.node]}")
                    else:
                        args.append(str(arg))
                print(", ".join(args), end="")
                print(f") : {_format_mlir_type(subcustom.type)}")
            print("  }")
        else:
            # Print other operations
            print(
                f"  %{node_names[node]} = {_convert_to_mlir_op(custom.tkw_op_name)}(",
                end="",
            )
            args = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    args.append(f"%{node_names[arg]}")
                elif isinstance(arg, fx.Proxy):
                    args.append(f"%{node_names[arg.node]}")
                else:
                    args.append(str(arg))
            print(", ".join(args), end="")
            print(f") : {_format_mlir_type(custom.type)}")

    print("  return None")
    print("}")
    print("-----")


def _convert_to_mlir_op(op_name: str) -> str:
    """
    Converts Python operation names to wave-style operation names.
    All operations are prefixed with 'wave.' to indicate they are wave operations.
    """
    op_map = {
        "add": "wave.add",
        "sub": "wave.sub",
        "mul": "wave.mul",
        "div": "wave.div",
        "read": "wave.read",
        "write": "wave.write",
        "extract": "wave.extract",
        "insert": "wave.insert",
        "broadcast": "wave.broadcast",
        "reshape": "wave.reshape",
        "shuffle": "wave.shuffle",
        "mma": "wave.mma",
        "sum": "wave.sum",
        "max": "wave.max",
        "min": "wave.min",
        "select": "wave.select",
        "cmp": "wave.cmp",
        "and": "wave.and",
        "or": "wave.or",
        "xor": "wave.xor",
        "not": "wave.not",
        "neg": "wave.neg",
        "abs": "wave.abs",
        "sqrt": "wave.sqrt",
        "exp": "wave.exp",
        "log": "wave.log",
        "sin": "wave.sin",
        "cos": "wave.cos",
        "tan": "wave.tan",
        "pow": "wave.pow",
        "floor": "wave.floor",
        "ceil": "wave.ceil",
        "round": "wave.round",
        "trunc": "wave.trunc",
        "fma": "wave.fma",
        "dot": "wave.dot",
        "matmul": "wave.matmul",
        "conv": "wave.conv",
        "reduce": "wave.reduce",
        "scan": "wave.scan",
        "mask": "wave.mask",
        "permute": "wave.permute",
        "shuffle": "wave.shuffle",
        "barrier": "wave.barrier",
        "sync": "wave.sync",
        "atomic": "wave.atomic",
        "return": "wave.return",
    }
    return op_map.get(op_name, f"wave.{op_name}")


def _format_mlir_type(type_obj):
    if type_obj is None:
        return "none"
    type_str = str(type_obj)
    if "tensor" in type_str or "vector" in type_str or "memref" in type_str:
        return type_str
    # Remove !stream.binding prefix and just return the inner type
    if "stream.binding" in type_str:
        return type_str.replace("!stream.binding<", "").replace(">", "")
    return type_str
