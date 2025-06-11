# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import logging
from iree.turbine.kernel.wave.scheduling.modulo_scheduling import (
    ModuloScheduler,
    EdgeWeight,
    Edge,
)
import torch.fx as fx
import numpy as np
import multiprocessing as mp
from iree.turbine.kernel.wave.visualization import visualize_graph
from iree.turbine.kernel.wave.scheduling.graph_utils import (
    find_strongly_connected_components,
    find_cycles_in_scc,
    all_pairs_longest_paths,
    evaluate_all_pairs_longest_paths,
)
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.wave.promotion import promote_placeholders
from iree.turbine.kernel.wave.hoisting import hoist_loop_invariant_ops
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from iree.turbine.kernel.wave.scheduling.schedule import schedule_graph
from iree.turbine.kernel.ops.wave_ops import get_custom
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.wave.utils.graph_utils import initialize_iter_args
from typing import Tuple, List, Dict
from iree.turbine.kernel.wave.scheduling.verifier import (
    ScheduleValidator,
)
from iree.turbine.kernel.wave.utils.print_utils import (
    load_schedule,
    parse_node_specs_from_schedule_file,
)
import os
from iree.turbine.kernel.wave.scheduling.resources import (
    resource_reservation_table,
    get_custom_operation_type,
    Operation,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.ops.wave_ops import Read, Write, MMA, IterArg

# Map node types from schedule file to custom operation classes
NODE_TYPE_TO_CUSTOM_OP = {
    Operation.READ_SHARED.value: (Read, "read"),
    Operation.WRITE_SHARED.value: (Write, "write"),
    Operation.READ_GLOBAL.value: (
        Read,
        "read",
    ),  # Global memory operations use same classes
    Operation.WRITE_GLOBAL.value: (Write, "write"),
    Operation.MMA.value: (MMA, "mma"),
    "IterArg": (
        IterArg,
        "placeholder",
    ),  # Special case since it's not in Operation enum
}

# Map schedule file node types to Operation enum values
SCHEDULE_NODE_TYPE_TO_OP = {
    "ReadShared": Operation.READ_SHARED.value,
    "WriteShared": Operation.WRITE_SHARED.value,
    "ReadGlobal": Operation.READ_GLOBAL.value,
    "WriteGlobal": Operation.WRITE_GLOBAL.value,
    "MMA": Operation.MMA.value,
}


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
        T = 4
        pool = mp.get_context("fork").Pool(processes=mp.cpu_count())
        D3 = all_pairs_longest_paths(graph, weighted_edges, T, pool)
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
        schedule, success = scheduler.schedule_graph()
        assert success == True
        assert schedule[nodes["a"]] == 0
        assert schedule[nodes["b"]] == 4
        assert schedule[nodes["c"]] == 5
        assert schedule[nodes["d"]] == 6
        assert scheduler.initiation_interval == 4
        assert np.all(
            scheduler.resource_reservations
            == np.array([[1, 1], [1, 0], [1, 0], [0, 1]])
        )

    def testGemmScheduling(self):

        # Input sizes
        M = tkl.sym.M
        N = tkl.sym.N
        K = tkl.sym.K
        # Workgroup tile sizes
        BLOCK_M = tkl.sym.BLOCK_M
        BLOCK_N = tkl.sym.BLOCK_N
        BLOCK_K = tkl.sym.BLOCK_K
        # Address space (for GPU, shared(1) or global(0))
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        # Other hyperparameters
        LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
        ARGK = tkl.sym.ARGK

        # Expose user-constraints
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]

        constraints += [
            tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
        ]

        @tkw.wave_trace_only(constraints)
        def gemm(
            a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        ):
            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

        hyperparams = {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            M: 2048,
            N: 10240,
            K: 1280,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 5,
            WRITE_GLOBAL_DELAY: 5,
            MMA_DELAY: 2,
            SHARED_MEMORY_UNITS: 2,
            GLOBAL_MEMORY_UNITS: 2,
            MMA_UNITS: 2,
            VALU_DELAY: 1,
            VALU_UNITS: 2,
            SHUFFLE_DELAY: 1,
            SHUFFLE_UNITS: 2,
        }
        with tk.gen.TestLaunchContext(hyperparams, canonicalize=True, schedule=True):
            trace: CapturedTrace = gemm()
            IndexingContext.current().finalize()
            initialize_iter_args(trace)
            add_get_results(trace)
            infer_types(trace)
            promote_placeholders(trace, constraints)
            hoist_loop_invariant_ops(trace, constraints)
            set_node_indices(trace, constraints)
            expand_graph(trace, constraints)
            set_post_expansion_indices(trace, constraints)
            minimize_global_loads(trace, constraints)
            schedule_graph(trace, constraints)
            subgraph = trace.get_subgraph("region_0")
            initiation_interval = 5
            correct_schedule = {
                "acc_1_1_0": {
                    "absolute_cycle": 10,
                    "cycle": 0,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "acc_1_0_0": {
                    "absolute_cycle": 10,
                    "cycle": 0,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "acc_0_1_0": {
                    "absolute_cycle": 9,
                    "cycle": 4,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_4": {
                    "absolute_cycle": 0,
                    "cycle": 0,
                    "stage": 0,
                    "initiation_interval": initiation_interval,
                },
                "write_2": {
                    "absolute_cycle": 5,
                    "cycle": 0,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_0_0": {
                    "absolute_cycle": 8,
                    "cycle": 3,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_0_1": {
                    "absolute_cycle": 7,
                    "cycle": 2,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_1_0_0": {
                    "absolute_cycle": 9,
                    "cycle": 4,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_1_0_1": {
                    "absolute_cycle": 9,
                    "cycle": 4,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_5": {
                    "absolute_cycle": 0,
                    "cycle": 0,
                    "stage": 0,
                    "initiation_interval": initiation_interval,
                },
                "write_3": {
                    "absolute_cycle": 5,
                    "cycle": 0,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_0_0": {
                    "absolute_cycle": 8,
                    "cycle": 3,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_0_1": {
                    "absolute_cycle": 7,
                    "cycle": 2,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_1_0": {
                    "absolute_cycle": 6,
                    "cycle": 1,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "read_shared_0_1_1": {
                    "absolute_cycle": 6,
                    "cycle": 1,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "mma_0_0_0": {
                    "absolute_cycle": 9,
                    "cycle": 4,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "mma_0_0_1": {
                    "absolute_cycle": 11,
                    "cycle": 1,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "mma_1_1_0": {
                    "absolute_cycle": 10,
                    "cycle": 0,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "mma_1_1_1": {
                    "absolute_cycle": 12,
                    "cycle": 2,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "mma_1_0_0": {
                    "absolute_cycle": 10,
                    "cycle": 0,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "mma_1_0_1": {
                    "absolute_cycle": 12,
                    "cycle": 2,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
                "mma_0_1_0": {
                    "absolute_cycle": 9,
                    "cycle": 4,
                    "stage": 1,
                    "initiation_interval": initiation_interval,
                },
                "mma_0_1_1": {
                    "absolute_cycle": 11,
                    "cycle": 1,
                    "stage": 2,
                    "initiation_interval": initiation_interval,
                },
            }
            for node in subgraph.nodes:
                custom = get_custom(node)
                if custom.name in correct_schedule:
                    assert custom.scheduling_parameters == correct_schedule[custom.name]


class ScheduleRepairerTest(unittest.TestCase):
    def create_test_graph(
        self,
    ) -> Tuple[fx.Graph, List[Edge], Dict[str, fx.Node]]:
        """Creates a simple test graph with resource requirements and dependencies."""
        graph = fx.Graph()
        target = lambda _: None

        # Create nodes with resource requirements
        a = graph.create_node("call_function", target, args=(), name="a")
        a.rrt = np.array([[1, 0]])  # Uses resource 0
        b = graph.create_node("call_function", target, args=(a,), name="b")
        b.rrt = np.array([[0, 1]])  # Uses resource 1
        c = graph.create_node("call_function", target, args=(b,), name="c")
        c.rrt = np.array([[1, 0]])  # Uses resource 0
        d = graph.create_node("call_function", target, args=(c,), name="d")
        d.rrt = np.array([[0, 1]])  # Uses resource 1

        # Create edges with latencies
        edges = [
            Edge(a, b, EdgeWeight(0, 2)),  # a->b with latency 2
            Edge(b, c, EdgeWeight(0, 1)),  # b->c with latency 1
            Edge(c, d, EdgeWeight(0, 1)),  # c->d with latency 1
        ]

        nodes = {"a": a, "b": b, "c": c, "d": d}
        return graph, edges, nodes

    def test_repair_backward(self):
        """Test moving a node that requires backward repair."""
        graph, edges, nodes = self.create_test_graph()
        T = 4  # Modulo scheduling period

        # Initial schedule with more spacing to allow for moves
        initial_schedule = {
            nodes["a"]: 0,  # Uses resource 0
            nodes["b"]: 3,  # Uses resource 1, a+2=2 but we put at 3 to allow moves
            nodes["c"]: 5,  # Uses resource 0, b+1=4 but we put at 5 to allow moves
            nodes["d"]: 6,  # Uses resource 1, c+1=6
        }

        validator = ScheduleValidator(
            initial_schedule=initial_schedule,
            T=T,
            nodes=list(nodes.values()),
            resource_limits=np.array([2, 2]),
            node_rrt_getter=lambda node: node.rrt,
            raw_edges_list=edges,
            num_resource_types=2,
        )

        # Try to move node 'c' to cycle 2 (which would require backward repair)
        success, new_schedule, _ = validator.attempt_move(nodes["c"], 2)

        # The move should succeed
        self.assertTrue(success, "Move should succeed")
        self.assertIsNotNone(new_schedule, "New schedule should not be None")

        # Verify that all dependencies are satisfied
        for edge in edges:
            u, v = edge._from, edge._to
            self.assertGreater(
                new_schedule[v],
                new_schedule[u],
                f"Dependency violation: {u.name}({new_schedule[u]}) -> {v.name}({new_schedule[v]})",
            )

        # Verify that node 'c' was moved to cycle 2
        self.assertEqual(
            new_schedule[nodes["c"]], 2, "Node 'c' should be moved to cycle 2"
        )

        # Verify that node 'b' was moved earlier to maintain dependency
        self.assertLess(
            new_schedule[nodes["b"]],
            new_schedule[nodes["c"]],
            f"Node 'b' should be scheduled before 'c' to maintain dependency",
        )

        # Verify that node 'a' remains at cycle 0
        self.assertEqual(
            new_schedule[nodes["a"]], 0, "Node 'a' should remain at cycle 0"
        )

        # Verify that node 'd' remains at cycle 6
        self.assertEqual(
            new_schedule[nodes["d"]], 6, "Node 'd' should remain at cycle 6"
        )

        # Verify resource constraints
        # Resource 0: a and c should not overlap
        self.assertNotEqual(
            new_schedule[nodes["a"]],
            new_schedule[nodes["c"]],
            "Nodes 'a' and 'c' using resource 0 should not overlap",
        )
        # Resource 1: b and d should not overlap
        self.assertNotEqual(
            new_schedule[nodes["b"]],
            new_schedule[nodes["d"]],
            "Nodes 'b' and 'd' using resource 1 should not overlap",
        )

    def test_repair_forward(self):
        """Test moving a node that requires forward repair."""
        graph, edges, nodes = self.create_test_graph()
        T = 4

        # Initial schedule with more spacing to allow for moves
        initial_schedule = {
            nodes["a"]: 0,  # Uses resource 0
            nodes["b"]: 2,  # Uses resource 1, a+2=2
            nodes["c"]: 5,  # Uses resource 0, b+1=3 but we put at 5 to allow moves
            nodes["d"]: 7,  # Uses resource 1, c+1=6 but we put at 7 to allow moves
        }

        validator = ScheduleValidator(
            initial_schedule=initial_schedule,
            T=T,
            nodes=list(nodes.values()),
            resource_limits=np.array([2, 2]),
            node_rrt_getter=lambda node: node.rrt,
            raw_edges_list=edges,
            num_resource_types=2,
        )

        # Try to move node 'b' to cycle 3 (which would require forward repair)
        success, new_schedule, _ = validator.attempt_move(nodes["b"], 3)

        # The move should succeed
        self.assertTrue(success)
        self.assertIsNotNone(new_schedule)

        # Verify that node 'b' was moved to cycle 3
        self.assertEqual(new_schedule[nodes["b"]], 3)

        # Verify that node 'c' was moved forward to maintain dependency
        self.assertGreater(new_schedule[nodes["c"]], new_schedule[nodes["b"]])

        # Verify that node 'd' was moved forward to maintain dependency
        self.assertGreater(new_schedule[nodes["d"]], new_schedule[nodes["c"]])

        # Verify all dependencies are satisfied
        for edge in edges:
            u, v = edge._from, edge._to
            self.assertGreater(
                new_schedule[v],
                new_schedule[u],
                f"Dependency violation: {u.name}({new_schedule[u]}) -> {v.name}({new_schedule[v]})",
            )

    def test_repair_failure(self):
        """Test a case where repair is impossible due to resource constraints."""
        graph, edges, nodes = self.create_test_graph()
        T = 4

        # Initial schedule with tight spacing to test resource constraints
        initial_schedule = {
            nodes["a"]: 0,  # Uses resource 0
            nodes["b"]: 2,  # Uses resource 1, a+2=2
            nodes["c"]: 3,  # Uses resource 0, b+1=3
            nodes["d"]: 4,  # Uses resource 1, c+1=4
        }

        validator = ScheduleValidator(
            initial_schedule=initial_schedule,
            T=T,
            nodes=list(nodes.values()),
            resource_limits=np.array(
                [1, 1]
            ),  # Reduced resource limits to force failure
            node_rrt_getter=lambda node: node.rrt,
            raw_edges_list=edges,
            num_resource_types=2,
        )

        # Try to move node 'c' to cycle 0 (conflicts with 'a' on resource 0)
        success, new_schedule, _ = validator.attempt_move(nodes["c"], 0)

        # The move should fail due to resource constraints
        self.assertFalse(success)
        self.assertIsNone(new_schedule)

    def test_load_and_move_schedule_txt(self):
        """Test loading a schedule from schedule.txt and moving a node."""
        self._test_load_and_move_schedule_txt_with_offset(1)
        self._test_load_and_move_schedule_txt_with_offset(-1)

    def _test_load_and_move_schedule_txt_with_offset(self, offset: int):
        """Helper method to test moving a node with a specific offset."""
        schedule_path = os.path.join(os.path.dirname(__file__), "schedule.txt")

        # Create the graph first with all nodes
        graph = fx.Graph()
        # Parse the schedule file to get node specifications
        node_specs = parse_node_specs_from_schedule_file(schedule_path)
        # Create nodes in the actual graph
        target = lambda *args, **kwargs: None
        for name, sort_key, node_type in node_specs:
            node = graph.create_node("call_function", target, args=(), name=name)
            node._sort_key = sort_key

        # Set resource requirements based on the node_type using the resource_reservation_table
        node_type_to_op = {
            "ReadShared": Operation.READ_SHARED,
            "WriteShared": Operation.WRITE_SHARED,
            "ReadGlobal": Operation.READ_GLOBAL,
            "WriteGlobal": Operation.WRITE_GLOBAL,
            "MMA": Operation.MMA,
            "IterArg": Operation.NOOP,
        }
        for (name, sort_key, node_type), node in zip(node_specs, graph.nodes):
            op_type = node_type_to_op.get(node_type, Operation.NOOP)
            node.rrt = resource_reservation_table[op_type]

        # Now load the schedule into our graph
        (
            schedule,
            initiation_interval,
            num_stages,
            nodes,
            edges,
            resource_reservations,
            resource_names,
        ) = load_schedule(schedule_path, graph)

        # Build a mapping from node to node_type
        node_to_type = {
            node: node_type
            for (name, sort_key, node_type), node in zip(node_specs, graph.nodes)
        }

        # Filter out edges that start with IterArg nodes
        filtered_edges = [
            edge for edge in edges if node_to_type.get(edge._from) != "IterArg"
        ]

        scheduling_params = get_default_scheduling_params()
        # Map resource names to their corresponding scheduling param values
        resource_name_to_param = {
            "GLOBAL_MEMORY_UNITS": GLOBAL_MEMORY_UNITS,
            "SHARED_MEMORY_UNITS": SHARED_MEMORY_UNITS,
            "MMA_UNITS": MMA_UNITS,
            "VALU_UNITS": VALU_UNITS,
            "SHUFFLE_UNITS": SHUFFLE_UNITS,
        }

        # Double the resource limits to allow more operations per cycle
        for param in resource_name_to_param.values():
            scheduling_params[param] *= 2

        validator = ScheduleValidator(
            initial_schedule=schedule,
            T=initiation_interval,
            nodes=nodes,
            resource_limits=np.array(
                [
                    scheduling_params[resource_name_to_param[name]]
                    for name in resource_names
                ]
            ),
            node_rrt_getter=lambda node: node.rrt,
            raw_edges_list=filtered_edges,  # Use filtered edges instead of original edges
            num_resource_types=5,
        )

        # Pick write_10 to move (sort key 5)
        # Move it backward by 1 cycle (from 2 to 1)
        node_to_move = next(node for node in nodes if node._sort_key == (5,))
        original_cycle = schedule[node_to_move]
        requested_cycle = original_cycle - 1 if offset < 0 else original_cycle + 1

        # Try to move the node
        success, new_schedule, _ = validator.attempt_move(node_to_move, requested_cycle)
        self.assertTrue(success, f"Failed to move node with offset {offset}")
        self.assertIsNotNone(new_schedule)

        # Verify the node was moved (either to requested cycle or a valid alternative)
        if offset < 0:
            if new_schedule[node_to_move] != original_cycle:
                self.assertLessEqual(
                    new_schedule[node_to_move],
                    requested_cycle,
                    f"Node moved too far backward with offset {offset}",
                )
            else:
                # If node couldn't be moved backward due to dependencies, that's valid
                pass
        else:
            self.assertNotEqual(
                new_schedule[node_to_move],
                original_cycle,
                f"Node was not moved from original cycle {original_cycle} with offset {offset}",
            )
            self.assertGreaterEqual(
                new_schedule[node_to_move],
                requested_cycle,
                f"Node not moved forward enough with offset {offset}",
            )

        # Verify all dependencies are satisfied (using filtered edges)
        for edge in filtered_edges:
            self.assertGreaterEqual(
                new_schedule[edge._to],
                new_schedule[edge._from] + edge.weight.delay,
                f"Dependency not met: {edge._from.name} -> {edge._to.name} with offset {offset}",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
