graphviz_disabled = False
try:
    import pygraphviz as pgv
except:
    graphviz_disabled = True
from torch import fx


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
