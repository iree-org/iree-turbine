# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from iree.turbine.kernel.ops.wave_ops import NestedRegionOp, Placeholder, get_custom
from ..._support.tracing import CapturedTrace
from typing import Sequence, Optional
import timeit


def print_graph(graph: fx.Graph):
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
    print(graph_str)


def print_trace(trace: CapturedTrace, custom_print: bool = True):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for name, subgraph in reversed(list(trace.region_graph.subgraphs.items())):
        if name == trace.root_graph:
            name = f"{name} [root]"
        print(f"{name}:\n")
        print_graph(subgraph)
        if custom_print:
            print("Custom format:")
            for node in subgraph.nodes:
                print(get_custom(node))


def print_subgraph(trace: CapturedTrace, subgraph_name: str, custom_print: bool = True):
    """
    Prints a specific subgraphs of a trace.
    The graphs are printed first in the torch printing format and
    then using our custom node format.
    """
    for name, subgraph in trace.region_graph.subgraphs.items():
        if name == subgraph_name:
            print(subgraph)
            if custom_print:
                for node in subgraph.nodes:
                    print(get_custom(node))


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
        print(f"Error in pass: {pass_name}\n")
        print_trace(trace)
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
