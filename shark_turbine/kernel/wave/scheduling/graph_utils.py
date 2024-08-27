import torch.fx as fx
from random import shuffle, seed
from collections import defaultdict


def find_strongly_connected_components(
    graph: fx.Graph, random_seed: int
) -> dict[fx.Node, list[fx.Node]]:
    """
    Find the strongly connected components in the graph.
    Returns a list of strongly connected components.
    """
    seed(random_seed)
    initial_times = list(range(len(graph.nodes)))
    shuffle(initial_times)
    for f, node in zip(initial_times, graph.nodes):
        node.f = f
    # Run DFS loop on reversed graph.
    run_dfs_loop(graph, 0, None, reverse=True)
    # Run DFS loop on original graph.
    run_dfs_loop(graph, 0, None, reverse=False)
    # Construct strongly connected components based on leaders.
    scc = defaultdict(list)
    for node in graph.nodes:
        scc[node.leader].append(node)
    return scc


def run_dfs_loop(graph: fx.Graph, t: int, s: fx.Node, reverse: bool) -> None:
    """
    Run the Depth First Search algorithm on the graph.
    """
    visited_nodes = set()
    sorted_nodes = sorted(graph.nodes, key=lambda x: x.f, reverse=True)
    for node in sorted_nodes:
        if node not in visited_nodes:
            s = node
            t, s = run_dfs(node, t, s, reverse, visited_nodes)


def run_dfs(
    node: fx.Node,
    t: int,
    s: fx.Node,
    reverse: bool,
    visited_nodes: set[fx.Node],
) -> None:
    """
    Run the Depth First Search algorithm on the graph.
    """
    visited_nodes.add(node)
    if not reverse:
        node.leader = s
    next_nodes = node.all_input_nodes if reverse else list(node.users.keys())
    for user in next_nodes:
        if user not in visited_nodes:
            t, s = run_dfs(user, t, s, reverse, visited_nodes)
    if reverse:
        t += 1
        node.f = t
    return t, s


def unblock(
    node: fx.Node, B: dict[fx.Node, list[fx.Node]], blocked: list[fx.Node]
) -> None:
    stack = set([node])
    while stack:
        candidate = stack.pop()
        if candidate in blocked:
            blocked.remove(candidate)
            stack.update(B[candidate])
            B[candidate].clear()


def circuit(node: fx.Node, B: dict[fx.Node, set[fx.Node]]) -> list[list[fx.Node]]:
    path = [node]
    stack = [(node, list(node.users.keys()))]
    blocked = set([node])
    circuits: list[list[fx.Node]] = []
    found_cycles = set()
    while stack:
        current, neighbors = stack[-1]
        if neighbors:
            candidate = neighbors.pop(0)
            if candidate == node:
                circuits.append(path + [candidate])
                found_cycles.update(path)
            elif not candidate in blocked:
                blocked.add(candidate)
                path.append(candidate)
                stack.append((candidate, list(candidate.users.keys())))
                found_cycles.discard(candidate)
                continue
        else:
            if current in found_cycles:
                unblock(current, B, blocked)
            else:
                for user in current.users:
                    B[user].add(current)
            stack.pop()
            path.pop()
    return circuits


def remove_node(graph: fx.Node, node: fx.Node) -> None:
    """
    Remove a node from the graph and update its users.
    """
    users = list(node.users.keys())
    for user in users:
        user.update_arg(user.args.index(node), None)
    graph.erase_node(node)


def find_cycles_in_graph(
    graph: fx.Graph, scc: dict[fx.Node, list[fx.Node]]
) -> list[list[fx.Node]]:
    """
    Find all simple cycles/circuits in the graph using Johnson's algorithm.
    """
    circuits: list[list[fx.Node]] = []
    B: dict[fx.Node, set[fx.Node]] = defaultdict(set)
    for _, nodes in scc.items():
        sorted_nodes = sorted(nodes, key=lambda x: x.f)
        while sorted_nodes:
            s = sorted_nodes.pop(0)
            for node in sorted_nodes:
                B[node] = set()
            circuits += circuit(s, B)
            remove_node(graph, s)
    return circuits
