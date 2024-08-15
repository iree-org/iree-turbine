disabled_graphviz = False
try:
    import pygraphviz as pgv
except:
    disabled_graphviz = True
from torch import fx


def visualize_graph(graph: fx.Graph, file_name: str):
    if disabled_graphviz:
        return
    G = pgv.AGraph(directed=True)
    for node in graph.nodes:
        G.add_node(node.name)
    for node in graph.nodes:
        for user in node.users.keys():
            G.add_edge(node.name, user.name)
    G.layout(prog="dot")
    G.draw(file_name)
