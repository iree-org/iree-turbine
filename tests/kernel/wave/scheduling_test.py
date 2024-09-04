import unittest
import logging
from shark_turbine.kernel.wave.scheduling.modulo_scheduling import (
    ModuloScheduler,
    EdgeWeight,
    Edge,
)
import torch.fx as fx
import numpy as np
from shark_turbine.kernel.wave.visualization import visualize_graph
from shark_turbine.kernel.wave.scheduling.graph_utils import (
    find_strongly_connected_components,
    find_cycles_in_scc,
    all_pairs_longest_paths,
    evaluate_all_pairs_longest_paths,
)


class SchedulingTest(unittest.TestCase):
    def create_graph_with_loops(self) -> fx.Graph:
        graph = fx.Graph()
        target = lambda _: None
        a = graph.create_node("call_function", target, args=(None,), name="a")
        b = graph.create_node(
            "call_function",
            target,
            args=(
                a,
                None,
                None,
            ),
            name="b",
        )
        c = graph.create_node("call_function", target, args=(b,), name="c")
        b.update_arg(1, c)
        d = graph.create_node("call_function", target, args=(c,), name="d")
        e = graph.create_node(
            "call_function",
            target,
            args=(
                c,
                d,
            ),
            name="e",
        )
        f = graph.create_node(
            "call_function",
            target,
            args=(
                e,
                a,
            ),
            name="f",
        )
        a.update_arg(0, c)
        b.update_arg(2, f)
        h = graph.create_node(
            "call_function",
            target,
            args=(b,),
            name="h",
        )
        i = graph.create_node(
            "call_function",
            target,
            args=(
                a,
                None,
            ),
            name="i",
        )
        j = graph.create_node(
            "call_function",
            target,
            args=(
                i,
                b,
            ),
            name="j",
        )
        i.update_arg(1, j)
        nodes = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "h": h, "i": i, "j": j}
        return graph, nodes

    def create_weighted_graph(self) -> fx.Graph:
        graph = fx.Graph()
        target = lambda _: None
        a = graph.create_node("call_function", target, args=(), name="a")
        a.rrt = np.array([[1, 0]])
        b = graph.create_node(
            "call_function",
            target,
            args=(
                a,
                None,
            ),
            name="b",
        )
        b.rrt = np.array([[0, 1]])
        c = graph.create_node("call_function", target, args=(b,), name="c")
        c.rrt = np.array([[1, 0]])
        d = graph.create_node(
            "call_function",
            target,
            args=(
                b,
                c,
            ),
            name="d",
        )
        d.rrt = np.array([[1, 0], [0, 1]])
        b.update_arg(1, d)
        edges = []
        edges.append(Edge(a, b, EdgeWeight(0, 2)))
        edges.append(Edge(b, c, EdgeWeight(0, 1)))
        edges.append(Edge(c, d, EdgeWeight(0, 1)))
        edges.append(Edge(d, b, EdgeWeight(1, 1)))
        edges.append(Edge(b, d, EdgeWeight(0, 2)))
        nodes = {"a": a, "b": b, "c": c, "d": d}
        return graph, edges, nodes

    def testGraphUtils(self):
        graph, nodes = self.create_graph_with_loops()
        visualize = False
        if visualize:
            visualize_graph(graph, "utils_test_graph.png")
        scc = find_strongly_connected_components(graph, 2024)
        assert len(scc) == 3
        expected_scc = {
            nodes["e"]: [
                nodes["a"],
                nodes["b"],
                nodes["c"],
                nodes["d"],
                nodes["e"],
                nodes["f"],
            ],
            nodes["h"]: [nodes["h"]],
            nodes["j"]: [nodes["i"], nodes["j"]],
        }
        for leader, scc_nodes in expected_scc.items():
            assert leader in scc
            assert scc[leader] == scc_nodes
        cycles = find_cycles_in_scc(scc)
        assert len(cycles) == 6
        expected_cycles = [
            [nodes["a"], nodes["b"], nodes["c"], nodes["a"]],
            [nodes["a"], nodes["f"], nodes["b"], nodes["c"], nodes["a"]],
            [nodes["b"], nodes["c"], nodes["b"]],
            [nodes["i"], nodes["j"], nodes["i"]],
            [nodes["f"], nodes["b"], nodes["c"], nodes["d"], nodes["e"], nodes["f"]],
            [nodes["f"], nodes["b"], nodes["c"], nodes["e"], nodes["f"]],
        ]
        for cycle in expected_cycles:
            assert cycle in cycles

    def testAPLP(self):
        graph, weighted_edges, nodes = self.create_weighted_graph()
        D = all_pairs_longest_paths(graph, weighted_edges)
        T = 4
        D3 = evaluate_all_pairs_longest_paths(D, T)
        assert D3[(nodes["a"], nodes["b"])] == 2
        assert D3[(nodes["a"], nodes["c"])] == 3
        assert D3[(nodes["a"], nodes["d"])] == 4
        assert D3[(nodes["b"], nodes["c"])] == 1
        assert D3[(nodes["b"], nodes["d"])] == 2
        assert D3[(nodes["c"], nodes["b"])] == 2 - T
        assert D3[(nodes["c"], nodes["d"])] == 1
        assert D3[(nodes["d"], nodes["b"])] == 1 - T
        assert D3[(nodes["d"], nodes["c"])] == 2 - T

    def testModuloScheduling(self):
        visualize = False
        graph, weighted_edges, nodes = self.create_weighted_graph()
        if visualize:
            visualize_graph(graph, "scheduling_test_graph.png")
        resources = np.array([1, 1])
        scheduler = ModuloScheduler(graph, weighted_edges, resources)
        schedule = scheduler.schedule()
        assert schedule[nodes["a"]] == 0
        assert schedule[nodes["b"]] == 4
        assert schedule[nodes["c"]] == 5
        assert schedule[nodes["d"]] == 6
        assert scheduler.initiation_interval == 4
        assert np.all(
            scheduler.resource_reservations
            == np.array([[1, 1], [1, 0], [1, 0], [0, 1]])
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
