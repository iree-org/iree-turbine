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

            @tkw.reduction(K, init_args=[c_reg])
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
