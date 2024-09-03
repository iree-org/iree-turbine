import torch.fx as fx
from random import shuffle, seed, Random
from collections import defaultdict
from ..._support.indexing import index_symbol, IndexExpr
from dataclasses import dataclass
import sympy
import math


@dataclass
class EdgeWeight:
    iteration_difference: int = 0
    delay: int = 0


@dataclass
class Edge:
    _from: fx.Node = None
    _to: fx.Node = None
    weight: EdgeWeight = None


def find_strongly_connected_components(
    graph: fx.Graph, random_seed: int
) -> dict[fx.Node, list[fx.Node]]:
    """
    Find the strongly connected components in the graph.
    Returns a list of strongly connected components.
    Uses Kosaraju's algorithm.
    References:
    [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
        Introduction to algorithms.
    """
    rng = Random(random_seed)
    initial_times = list(range(len(graph.nodes)))
    rng.shuffle(initial_times)
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
) -> tuple[int, fx.Node]:
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
    References:
    [1] Johnson, Donald B. "Finding all the elementary circuits of a directed graph."
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


def all_pairs_longest_paths(
    graph: fx.Graph,
    edges: list[Edge],
) -> dict[tuple[int, int], IndexExpr]:
    """
    For each node in the graph, compute the longest path to all other nodes.
    Uses the Floyd-Warshall algorithm and assumes that the cycles don't
    have positive weights.
    """
    T = index_symbol("$INITIATION_INTERVAL")
    D: dict[tuple[int, int, int]] = {}
    for v in graph.nodes:
        for w in graph.nodes:
            D[(v, w)] = -math.inf
    for edge in edges:
        D[(edge._from, edge._to)] = (
            edge.weight.delay - edge.weight.iteration_difference * T
        )
    for u in graph.nodes:
        for v in graph.nodes:
            for w in graph.nodes:
                D[(v, w)] = sympy.Max(D[(v, w)], D[(v, u)] + D[(u, w)])
    return D


def evaluate_all_pairs_longest_paths(
    D: dict[tuple[int, int], IndexExpr], initiation_interval: int
) -> dict[tuple[int, int], int]:
    """
    Substitute the initiation interval into the longest paths.
    """
    T = index_symbol("$INITIATION_INTERVAL")
    for key in D:
        D[key] = D[key].subs(T, initiation_interval)
    return D
