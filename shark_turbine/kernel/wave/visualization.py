# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

graphviz_disabled = False
try:
    import pygraphviz as pgv
except:
    graphviz_disabled = True
from torch import fx
from .scheduling.graph_utils import Edge
import math


def number_nodes(graph: fx.Graph) -> dict[int, int]:
    return {id(node): i for i, node in enumerate(graph.nodes)}


def visualize_graph(graph: fx.Graph, file_name: str):
    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    node_numbering = number_nodes(graph)
    G = pgv.AGraph(directed=True)
    for node in graph.nodes:
        G.add_node(node_numbering[id(node)], label=node.name)
    for node in graph.nodes:
        for user in node.users.keys():
            G.add_edge(node_numbering[id(node)], node_numbering[id(user)])
    G.layout(prog="dot")
    G.draw(file_name)


def visualize_edges(edges: list[Edge], file_name: str):
    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    G = pgv.AGraph(directed=True)
    node_map = {}
    count = 0
    for edge in edges:
        if edge._from not in node_map:
            node_map[edge._from] = count
            count += 1
            G.add_node(node_map[edge._from], label=f"{edge._from}")
        if edge._to not in node_map:
            node_map[edge._to] = count
            count += 1
            G.add_node(node_map[edge._to], label=f"{edge._to}")
        G.add_edge(
            node_map[edge._from],
            node_map[edge._to],
            label=f"({edge.weight.iteration_difference}, {edge.weight.delay})",
        )
    G.layout(prog="dot")
    G.draw(file_name)


def visualize_schedule(
    schedule: dict[fx.Graph, int], initiation_interval: int, file_name: str
):
    import pandas as pd

    max_time = max(schedule.values())
    max_stage = math.ceil(max_time / initiation_interval)
    rows = max_time + 1 + max_stage * initiation_interval
    cols = max_stage

    table = [["" for _ in range(cols)] for _ in range(rows)]
    for stage in range(max_stage):
        for key, value in schedule.items():
            table[value + stage * initiation_interval][stage] += f"{key}<br>"

    df = pd.DataFrame(table, columns=[f"Stage {i}" for i in range(cols)])
    s = df.style.set_properties(**{"text-align": "center"})
    s = s.set_table_styles(
        [
            {"selector": "", "props": [("border", "1px solid grey")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("min-width", "300px")]},
        ]
    )
    output = s.apply(
        lambda x: [
            (
                "background: lightgreen"
                if int(x.name) >= (max_stage - 1) * initiation_interval
                and int(x.name) < max_stage * initiation_interval
                else ""
            )
            for _ in x
        ],
        axis=1,
    ).to_html()
    with open(f"{file_name}", "w") as f:
        f.write(output)
