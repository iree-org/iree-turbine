# RUN: python %s | FileCheck %s

import logging
from typing import Callable
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.promotion import promote_node
from shark_turbine.kernel.wave.hoisting import hoist_allocs
from shark_turbine.kernel.wave.address_spaces import *
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel._support.indexing import IndexingContext
from shark_turbine.kernel.ops.wave_ops import *


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()
        # Print a separator between tests
        print("-----")
    return func


def get_read_nodes(graph: fx.Graph) -> list[fx.Node]:
    nodes: list[fx.Node] = list(graph.nodes)
    return [node for node in nodes if hasattr(node, "tkw_op") and node.tkw_op == Read]


def print_trace(trace: CapturedTrace):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and then using
    our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for subgraph in reversed(list(trace.region_graph.subgraphs.values())):
        print(subgraph)


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


@tkw.wave_trace_only()
def read_write_same_size(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run
def test_read_write_equal_sizes():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        }
    ):
        trace: CapturedTrace = read_write_same_size()
        graph: fx.Graph = trace.get_root_graph()
        read_node = get_read_nodes(graph)[0]
        IndexingContext.current().finalize()
        promote_node(read_node, graph, SHARED_ADDRESS_SPACE)
        print_trace(trace)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_1_1
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_1_0
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_0_1
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %write_0_0
        # CHECK-SAME: (%read, %c, 4)
        # CHECK-NEXT: %write_1_1
        # CHECK-SAME: (%read_1_1, %c, 4)
        # CHECK-NEXT: %write_1_0
        # CHECK-SAME: (%read_1_0, %c, 4)
        # CHECK-NEXT: %write_0_1
        # CHECK-SAME: (%read_0_1, %c, 4)

        # CHECK: -----


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
        }
    ):
        trace: CapturedTrace = gemm()
        graph: fx.Graph = trace.get_subgraph("region_0")
        read_nodes = get_read_nodes(graph)
        for read_node in read_nodes:
            promote_node(read_node, graph, SHARED_ADDRESS_SPACE)
        hoist_allocs(trace)
        IndexingContext.current().finalize()
        print_trace(trace)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_0_0_0
        # CHECK-NEXT: %register_1_1_0
        # CHECK-NEXT: %register_1_0_0
        # CHECK-NEXT: %register_0_1_0
        # CHECK-NEXT: %reduction
        # CHECK-SAME: %register_0_0_0, %register_0_1_0, %register_1_0_0, %register_1_1_0
        # CHECK-NEXT: %getresult_1_1_0
        # CHECK-NEXT: %getresult_1_0_0
        # CHECK-NEXT: %getresult_0_1_0
        # CHECK-NEXT: %getresult_0_0_0
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%get_result_0_0_0, %c, 4)
        # CHECK-NEXT: %write_1_1_0
        # CHECK-SAME: (%get_result_1_1_0, %c, 4)
        # CHECK-NEXT: %write_1_0_0
        # CHECK-SAME: (%get_result_1_0_0, %c, 4)
        # CHECK-NEXT: %write_0_1_0
        # CHECK-SAME: (%get_result_0_1_0, %c, 4)

        # Reduction subgraph:

        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_1_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_0_1_0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_1_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK-NEXT: %read_1_0_1
        # CHECK-SAME: (%a, 4)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%b, 4)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%b, 4)
        # CHECK-NEXT: %read_0_1_0
        # CHECK-SAME: (%b, 4)
        # CHECK-NEXT: %read_0_1_1
        # CHECK-SAME: (%b, 4)

        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0, %acc)
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-SAME: (%read_0_0_1, %read_0_0_1, %mma_0_0_0)
        # CHECK-NEXT: %mma_1_1_0
        # CHECK-SAME: (%read_1_0_0, %read_0_1_0, %acc_1_1_0)
        # CHECK-NEXT: %mma_1_1_1
        # CHECK-SAME: (%read_1_0_1, %read_0_1_1, %mma_1_1_0)
        # CHECK-NEXT: %mma_1_0_0
        # CHECK-SAME: (%read_1_0_0, %read_0_0_0, %acc_1_0_0)
        # CHECK-NEXT: %mma_1_0_1
        # CHECK-SAME: (%read_1_0_1, %read_0_0_1, %mma_1_0_0)
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-SAME: (%read_0_0_0, %read_0_1_0, %acc_0_1_0)
        # CHECK-NEXT: %mma_0_1_1
        # CHECK-SAME: (%read_0_0_1, %read_0_1_1, %mma_0_1_0)
        # CHECK-NEXT: return [mma_0_0_1, mma_1_1_1, mma_1_0_1, mma_0_1_1]

        # CHECK-NEXT: -----


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
