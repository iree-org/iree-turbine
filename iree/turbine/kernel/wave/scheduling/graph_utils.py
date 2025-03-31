# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.fx as fx
from random import Random
from collections import defaultdict, deque
from ..._support.indexing import index_symbol, IndexExpr
from .resources import *
from dataclasses import dataclass
import sympy
import math
from functools import partial
from ..utils.symbol_utils import safe_subs
import multiprocessing as mp
from typing import Optional, Callable

T = index_symbol("$INITIATION_INTERVAL")


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


def circuit(
    node: fx.Node, B: dict[fx.Node, set[fx.Node]], explored: set[fx.Node]
) -> list[list[fx.Node]]:
    path = [node]
    stack = [(node, list(node.users.keys()))]
    blocked = set([node])
    circuits: list[list[fx.Node]] = []
    found_cycles = set()
    while stack:
        current, neighbors = stack[-1]
        if neighbors:
            candidate = neighbors.pop(0)
            if candidate in explored:
                continue
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


def find_cycles_in_scc(scc: dict[fx.Node, list[fx.Node]]) -> list[list[fx.Node]]:
    """
    Find all simple cycles/circuits in the graph using Johnson's algorithm.
    References:
    [1] Johnson, Donald B. "Finding all the elementary circuits of a directed graph."
    """
    circuits: list[list[fx.Node]] = []
    B: dict[fx.Node, set[fx.Node]] = defaultdict(set)
    for _, nodes in scc.items():
        sorted_nodes = sorted(nodes, key=lambda x: x.f)
        explored = set()
        while sorted_nodes:
            s = sorted_nodes.pop(0)
            for node in sorted_nodes:
                B[node] = set()
            circuits += circuit(s, B, explored)
            explored.add(s)
    return circuits


def all_pairs_longest_paths_helper(
    graph: fx.Graph, u: fx.Node, dist: dict[tuple[fx.Node, fx.Node], IndexExpr], i: int
):
    v = list(graph.nodes)[i]
    for w in graph.nodes:
        dist[(v, w)] = sympy.Max(dist[(v, w)], dist[(v, u)] + dist[(u, w)])
    return v, dist


def all_pairs_longest_path_parallel(N: int, D: np.array, k: int, i: int):
    """
    This function is called once for a different value of i.
    """
    for j in range(N):
        D[i, j] = np.maximum(D[i, j], D[i, k] + D[k, j])
    return i, D[i]


def all_pairs_longest_paths_symbolic(
    graph: fx.Graph,
    edges: list[Edge],
) -> dict[tuple[fx.Node, fx.Node], IndexExpr]:
    """
    For each node in the graph, compute the longest path to all other nodes.
    Uses the Floyd-Warshall algorithm and assumes that the cycles don't
    have positive weights.
    """
    D: dict[tuple[fx.Node, fx.Node], int] = {}
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


def all_pairs_longest_paths(
    graph: fx.Graph, edges: list[Edge], T: int, pool: Optional[mp.Pool]
) -> dict[tuple[fx.Node, fx.Node], IndexExpr]:
    """
    For each node in the graph, compute the longest path to all other nodes.
    Uses the Floyd-Warshall algorithm and assumes that the cycles don't
    have positive weights. This function computes the distances in parallel
    by parallelizing across the start nodes.

    T is the initiation interval that is computed during modulo scheduling.
    """
    N = len(graph.nodes)
    D = np.zeros((N, N), dtype=np.float32)
    negative_inf = -np.inf
    for i in range(N):
        for j in range(N):
            D[i, j] = negative_inf

    all_nodes = list(graph.nodes)
    for edge in edges:
        i = all_nodes.index(edge._from)
        j = all_nodes.index(edge._to)
        D[i, j] = edge.weight.delay - edge.weight.iteration_difference * T

    # Parallel implementation
    for k in range(N):
        func = partial(all_pairs_longest_path_parallel, N, D, k)
        if pool is not None:
            results = pool.map(func, range(N))
        else:
            results = map(func, range(N))

        for result in results:
            D[result[0]] = result[1]

    # Convert from index to node based representation.
    G: dict[tuple[fx.Node, fx.Node], int] = {}
    for i, from_node in enumerate(graph.nodes):
        for j, to_node in enumerate(graph.nodes):
            if np.isinf(D[i, j]) or i == j:
                continue
            G[(from_node, to_node)] = int(D[i, j])

    return G


def evaluate_all_pairs_longest_paths(
    D: dict[tuple[fx.Node, fx.Node], IndexExpr], initiation_interval: int
) -> dict[tuple[fx.Node, fx.Node], int]:
    """
    Substitute the initiation interval into the longest paths. Remove
    any negative infinity values.
    """
    D_static = dict(D)
    for key in D_static:
        D_static[key] = safe_subs(D_static[key], [(T, initiation_interval)])
    # Remove the negative infinity values and edges to self.
    for k in list(D_static.keys()):
        if math.isinf(D_static[k]) or k[0] == k[1]:
            del D_static[k]
    return D_static


def topological_sort(scc: dict[fx.Node, list[fx.Node]]) -> dict[fx.Node, list[fx.Node]]:
    """
    Perform a topological sort on the strongly connected components.
    """
    sorted_keys = sorted(scc.keys(), key=lambda x: x.f)
    return {k: scc[k] for k in sorted_keys}


def topological_sort_nodes(
    scc: list[fx.Node], edges: list[Edge], exclude: list[fx.Node] = None
) -> list[fx.Node]:
    """
    Perform a topological sort on the nodes in the strongly connected component that have an edge in edges, excluding
    certain nodes.
    """
    scc_nodes = set(scc)
    filtered_nodes = set()
    for edge in edges:
        if edge._from in scc_nodes and edge._to in scc_nodes:
            filtered_nodes.add(edge._to)
            filtered_nodes.add(edge._from)
    filtered_nodes -= set(exclude) if exclude is not None else set()
    sorted_nodes = sorted(filtered_nodes, key=lambda x: x.f)
    return sorted_nodes


def get_scheduling_weight(node: fx.Node) -> EdgeWeight:
    """
    Get the scheduling weight of a node.
    """
    custom_node = get_custom(node)
    match custom_node:
        case Read():
            if custom_node.memory_type.address_space == GLOBAL_ADDRESS_SPACE:
                weight = EdgeWeight(0, delay_table[Operation.READ_GLOBAL])
            else:
                weight = EdgeWeight(0, delay_table[Operation.READ_SHARED])
        case Write():
            if custom_node.memory_type.address_space == GLOBAL_ADDRESS_SPACE:
                weight = EdgeWeight(0, delay_table[Operation.WRITE_GLOBAL])
            else:
                weight = EdgeWeight(0, delay_table[Operation.WRITE_SHARED])
        case MMA():
            weight = EdgeWeight(0, delay_table[Operation.MMA])
        case IterArg():
            weight = EdgeWeight(1, 0)
        case CastOp() | Extract() | Permute() | Broadcast() | Reshape() | SelfIndex():
            weight = EdgeWeight(0, delay_table[Operation.NOOP])
        case ApplyExpr() | UnaryPyOp() | BinaryOpBase() | SelectOp():
            weight = EdgeWeight(0, delay_table[Operation.VALU])
        case ShuffleOp():
            weight = EdgeWeight(0, delay_table[Operation.SHUFFLE])
        case _:
            raise ValueError(f"Unsupported node type: {custom_node}")
    weight.delay = subs_idxc(weight.delay)
    weight.iteration_difference = subs_idxc(weight.iteration_difference)
    return weight


def erase_placeholder_nodes(graph: fx.Graph, ignore_nodes: set[fx.Node]) -> None:
    """
    This function erases nodes in the ignore list from the graph. We replace uses
    of the node with None.
    """
    for node in ignore_nodes:
        for user in list(node.users):
            idx = user.args.index(node)
            user.update_arg(idx, None)
        graph.erase_node(node)


def create_scheduling_edges(
    graph: fx.Graph,
    ignore_nodes: set[fx.Node],
    iter_args: list[fx.Node],
    output: fx.Node,
) -> list[Edge]:
    """
    Create scheduling edges from the graph including back edges
    from the outputs to the iter args. Also remove output
    and placeholder nodes.
    """
    # Create edges from outputs to iter args.
    for return_val, iter_arg in zip(get_custom(output).return_vals[0], iter_args):
        iter_arg.args = (return_val,)
    graph.erase_node(output)
    edges = []
    for node in graph.nodes:
        if node in ignore_nodes:
            continue
        weight = get_scheduling_weight(node)
        for user in node.users:
            edge = Edge(node, user, weight)
            edges.append(edge)
    erase_placeholder_nodes(graph, ignore_nodes)
    return edges


def get_root_nodes_from_edges(edges: list[Edge]) -> list[fx.Node]:
    """
    Given scheduling edges, returns a list of node that
    do not have a ancestor, i.e are root nodes of the graph.
    """
    source_nodes = set()
    dst_nodes = set()
    for edge in edges:
        source_nodes.add(edge._from)
        dst_nodes.add(edge._to)
    root_nodes = source_nodes.difference(dst_nodes)
    return root_nodes


def filter_edges(filter: Callable[[Edge], bool], edges) -> list[Edge]:
    filtered = []
    for edge in edges:
        if filter(edge):
            filtered.append(edge)
    return filtered


def sort_graph_by_edge_weight(nodes: list[fx.Node], edges: list[Edge]):
    """
    Sort nodes based on scheduling weight.

    Where scheduling weight is defined as:
    scheduling_weight[node] = sum(edge.weight + scheduling_weight[edge.source] for edge in edges)

    scheduling weight is suppose to quantify how early or late a node should live in a graph.
    Similar to topoological ordering but more robust.

    This is achieved by setting up a workqueue which starts with root of graph.
    we then explore successors of nodes in workqueue and compute it's scheduling
    weight as defined in above formula. If ancestor/producer to current node has
    no value yet, we move current node to end of queue to try again later.
    """
    schedule_weight = {}
    workqueue = deque(nodes)
    non_solved_counter = 0
    while len(workqueue) > 0:
        node = workqueue.popleft()
        is_producer_edge = lambda edge: edge._to == node
        producers_edges = filter_edges(is_producer_edge, edges)
        filter_producer_edge = [
            edge for edge in producers_edges if edge.weight.iteration_difference == 0
        ]

        # Save for later if producer not registered yet.
        if any([edge._from not in schedule_weight for edge in filter_producer_edge]):
            # If we went over entire workqueue and still cannot find producer,
            # means it is missing producer from the edges.
            non_solved_counter += 1
            if non_solved_counter > len(workqueue):
                raise ValueError(
                    "Cannot find producer(s) for remaining item in workqueue."
                )
            workqueue.append(node)
            continue

        non_solved_counter = 0
        schedule_weight[node] = sum(
            [
                schedule_weight[edge._from] + edge.weight.delay
                for edge in filter_producer_edge
            ]
        )
    return sorted(nodes, key=lambda x: schedule_weight[x])
