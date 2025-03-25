# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from ..._support.tracing import CapturedTrace
from ...ops.wave_ops import get_custom
from typing import Sequence


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
):
    if "all" in print_ir_before or p.__name__ in print_ir_before:
        print(f"***Before {p.__name__}***\n")
        print_trace(trace)
    try:
        p()
    except Exception:
        print(f"Error in pass: {p.__name__}\n")
        print_trace(trace)
        raise
    if "all" in print_ir_after or p.__name__ in print_ir_after:
        print(f"***After {p.__name__}***\n")
        print_trace(trace)
