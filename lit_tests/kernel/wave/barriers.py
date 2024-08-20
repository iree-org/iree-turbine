# RUN: python %s | FileCheck %s

import logging
from typing import Callable
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.promotion import promote_node, promote_placeholders
from shark_turbine.kernel.wave.barriers import add_shared_memory_barriers
from shark_turbine.kernel.wave.hoisting import hoist_allocs
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel._support.indexing import IndexingContext
from shark_turbine.kernel.ops.wave_ops import *
from shark_turbine.kernel.wave.utils import run_test, print_trace


def get_read_nodes(graph: fx.Graph) -> list[CustomOp]:
    custom_nodes: list[CustomOp] = [get_custom(node) for node in graph.nodes]
    return [node for node in custom_nodes if isinstance(node, Read)]


def tweak_index(graph: fx.Graph):
    promoted_read_nodes = [
        node for node in get_read_nodes(graph) if node.write_dependency is not None
    ]
    # Modify the write dependency index to trigger a barrier.
    for promoted_read_node in promoted_read_nodes:
        write_dependency = promoted_read_node.write_dependency
        for key, value in write_dependency.index.items():
            write_dependency.index[key].start = value.start + 1


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
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
ADDRESS_SPACE_1 = tkl.sym.ADDRESS_SPACE_1

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def read_write_same_size(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def test_read_write_equal_sizes():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
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
        promote_node(read_node, SHARED_ADDRESS_SPACE, constraints)
        expand_graph(trace, constraints)
        tweak_index(graph)
        add_shared_memory_barriers(trace)
        print_trace(trace, False)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_1
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_1
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %allocate
        # CHECK-SAME: ((M, N), (BLOCK_M, BLOCK_N), f16, $SHARED_ADDRESS_SPACE)
        # CHECK-NEXT: %write_shared_0_0
        # CHECK-SAME: (%read_0_0, %allocate, 4, None)
        # CHECK-NEXT: %write_shared_1_1
        # CHECK-SAME: (%read_1_1, %allocate, 4, None)
        # CHECK-NEXT: %write_shared_1_0
        # CHECK-SAME: (%read_1_0, %allocate, 4, None)
        # CHECK-NEXT: %write_shared_0_1
        # CHECK-SAME: (%read_0_1, %allocate, 4, None)
        # CHECK-NEXT: %barrier
        # CHECK-NEXT: %read_shared_0_0
        # CHECK-SAME: (%allocate, 4, None, %write_shared_0_0)
        # CHECK-NEXT: %read_shared_1_1
        # CHECK-SAME: (%allocate, 4, None, %write_shared_1_1)
        # CHECK-NEXT: %read_shared_1_0
        # CHECK-SAME: (%allocate, 4, None, %write_shared_1_0)
        # CHECK-NEXT: %read_shared_0_1
        # CHECK-SAME: (%allocate, 4, None, %write_shared_0_1)
        # CHECK-NEXT: %write_0_0
        # CHECK-SAME: (%read_shared_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_1
        # CHECK-SAME: (%read_shared_1_1, %c, 4, None)
        # CHECK-NEXT: %write_1_0
        # CHECK-SAME: (%read_shared_1_0, %c, 4, None)
        # CHECK-NEXT: %write_0_1
        # CHECK-SAME: (%read_shared_0_1, %c, 4, None)
        # CHECK-NEXT: return None

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


@run_test
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
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
        IndexingContext.current().finalize()
        read_nodes = get_read_nodes(graph)
        for read_node in read_nodes:
            promote_node(read_node, SHARED_ADDRESS_SPACE, constraints)
        hoist_allocs(trace)
        expand_graph(trace, constraints)
        tweak_index(graph)
        add_shared_memory_barriers(trace)
        print_trace(trace, False)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_0_0_0
        # CHECK-NEXT: %register_1_1_0
        # CHECK-NEXT: %register_1_0_0
        # CHECK-NEXT: %register_0_1_0
        # CHECK-NEXT: %allocate
        # CHECK-SAME: ((M, K), (BLOCK_M, K), f16, $SHARED_ADDRESS_SPACE)
        # CHECK-NEXT: %allocate_1
        # CHECK-SAME: ((N, K), (BLOCK_N, K), f16, $SHARED_ADDRESS_SPACE)
        # CHECK-NEXT: reduction
        # CHECK-SAME (K, [%register_0_0_0, %register_1_1_0, %register_1_0_0, %register_0_1_0]
        # CHECK-NEXT: %getresult_1_1_0
        # CHECK-SAME: (%reduction, 3)
        # CHECK-NEXT: %getresult_1_0_0
        # CHECK-SAME: (%reduction, 2)
        # CHECK-NEXT: %getresult_0_1_0
        # CHECK-SAME: (%reduction, 1)
        # CHECK-NEXT: %getresult_0_0_0
        # CHECK-SAME: (%reduction, 0)
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%getresult_0_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_1_0
        # CHECK-SAME: (%getresult_1_1_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_0
        # CHECK-SAME: (%getresult_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_0_1_0
        # CHECK-SAME: (%getresult_0_1_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Reduction subgraph:
        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_1_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_0_1_0
        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_0_0_0
        # CHECK-NEXT: %read_0_0_1
        # CHECK-NEXT: %read_1_0_0
        # CHECK-NEXT: %read_1_0_1
        # CHECK-NEXT: %write_shared_0_0_0
        # CHECK-NEXT: %write_shared_0_0_1
        # CHECK-NEXT: %write_shared_1_0_0
        # CHECK-NEXT: %write_shared_1_0_1
        # CHECK-NEXT: %barrier
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_shared_1_0_0
        # CHECK-NEXT: %read_shared_1_0_1
        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_0_0_0
        # CHECK-NEXT: %read_0_0_1
        # CHECK-NEXT: %read_0_1_0
        # CHECK-NEXT: %read_0_1_1
        # CHECK-NEXT: %write_shared_0_0_0
        # CHECK-NEXT: %write_shared_0_0_1
        # CHECK-NEXT: %write_shared_0_1_0
        # CHECK-NEXT: %write_shared_0_1_1
        # CHECK-NEXT: %barrier_1
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_shared_0_1_0
        # CHECK-NEXT: %read_shared_0_1_1
        # CHECK-NEXT: %mma_0_0_0
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-NEXT: %mma_1_1_0
        # CHECK-NEXT: %mma_1_1_1
        # CHECK-NEXT: %mma_1_0_0
        # CHECK-NEXT: %mma_1_0_1
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-NEXT: %mma_0_1_1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
