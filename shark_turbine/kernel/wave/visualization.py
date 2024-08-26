graphviz_disabled = False
try:
    import pygraphviz as pgv
except:
    graphviz_disabled = True
from torch import fx
import warnings


def visualize_graph(graph: fx.Graph, file_name: str):
    if graphviz_disabled:
        raise ImportError("pygraphviz not installed, cannot visualize graph")
    G = pgv.AGraph(directed=True)
    for node in graph.nodes:
        G.add_node(id(node), label=node.name)
    for node in graph.nodes:
        for user in node.users.keys():
            G.add_edge(id(node), id(user))
    G.layout(prog="dot")
    G.draw(file_name)
