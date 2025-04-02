# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.wave.promotion import promote_node, promote_placeholders
from iree.turbine.kernel.wave.barriers import add_shared_memory_barriers
from iree.turbine.kernel.wave.hoisting import hoist_loop_invariant_ops
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.ops.wave_ops import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.print_utils import (
    print_trace,
)
from iree.turbine.kernel.wave.utils.graph_utils import (
    initialize_iter_args,
)


def get_read_nodes(graph: fx.Graph) -> list[CustomOp]:
    custom_nodes: list[CustomOp] = [get_custom(node) for node in graph.nodes]
    return [node for node in custom_nodes if isinstance(node, Read)]


def tweak_index(graph: fx.Graph):
    promoted_read_nodes = [
        node for node in get_read_nodes(graph) if node.write_dependency is not None
    ]
    # Modify the write dependency index to trigger a barrier.
    for promoted_read_node in promoted_read_nodes:
        write_dependency = promoted_read_node.write_dependency[0]
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
        add_get_results(trace)
        infer_types(trace)
        promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        tweak_index(graph)
        add_shared_memory_barriers(trace)
        print_trace(trace, False)
        # CHECK: %allocate
        # CHECK-SAME: ((M, N), (BLOCK_M, BLOCK_N + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_M:0_N:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:0_N:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %write_1_shared_M:0_N:0
        # CHECK-SAME: (%read_M:0_N:0, %allocate, 4, None, ())
        # CHECK-NEXT: %write_1_shared_M:0_N:1
        # CHECK-SAME: (%read_M:0_N:1, %allocate, 4, None, ())
        # CHECK-NEXT: %write_1_shared_M:1_N:0
        # CHECK-SAME: (%read_M:1_N:0, %allocate, 4, None, ())
        # CHECK-NEXT: %write_1_shared_M:1_N:1
        # CHECK-SAME: (%read_M:1_N:1, %allocate, 4, None, ())
        # CHECK-NEXT: %shared_memory_barrier
        # CHECK-NEXT: %read_1_shared_M:0_N:0
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1_shared_M:0_N:0]
        # CHECK-NEXT: %read_1_shared_M:0_N:1
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1_shared_M:0_N:1]
        # CHECK-NEXT: %read_1_shared_M:1_N:0
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1_shared_M:1_N:0]
        # CHECK-NEXT: %read_1_shared_M:1_N:1
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1_shared_M:1_N:1]
        # CHECK-NEXT: %write_M:0_N:0
        # CHECK-SAME: (%read_1_shared_M:0_N:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:1
        # CHECK-SAME: (%read_1_shared_M:0_N:1, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0
        # CHECK-SAME: (%read_1_shared_M:1_N:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:1
        # CHECK-SAME: (%read_1_shared_M:1_N:1, %c, 4, None, ())
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
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        read_nodes = get_read_nodes(graph)
        for read_node in read_nodes:
            promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        tweak_index(graph)
        hoist_loop_invariant_ops(trace, constraints)
        add_shared_memory_barriers(trace)
        print_trace(trace, False)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_M:0_N:0_K:0
        # CHECK-NEXT: %register_M:0_N:1_K:0
        # CHECK-NEXT: %register_M:1_N:0_K:0
        # CHECK-NEXT: %register_M:1_N:1_K:0
        # CHECK-NEXT: %allocate_1
        # CHECK-SAME: ((N, K), (BLOCK_N, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: %allocate
        # CHECK-SAME: ((M, K), (BLOCK_M, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: reduction
        # CHECK-SAME: (K, [%register_M:0_N:0_K:0, %register_M:0_N:1_K:0, %register_M:1_N:0_K:0, %register_M:1_N:1_K:0]
        # CHECK-NEXT: %get_result_M:0_N:0_K:0
        # CHECK-SAME: (%reduction, 0)
        # CHECK-NEXT: %get_result_M:0_N:1_K:0
        # CHECK-SAME: (%reduction, 1)
        # CHECK-NEXT: %get_result_M:1_N:0_K:0
        # CHECK-SAME: (%reduction, 2)
        # CHECK-NEXT: %get_result_M:1_N:1_K:0
        # CHECK-SAME: (%reduction, 3)
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:1_K:0
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:1_K:0
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # Reduction subgraph:
        # CHECK: %b
        # CHECK: %a
        # CHECK-NEXT: %acc_M:0_N:0_K:0
        # CHECK-NEXT: %acc_M:0_N:1_K:0
        # CHECK-NEXT: %acc_M:1_N:0_K:0
        # CHECK-NEXT: %acc_M:1_N:1_K:0
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-NEXT: %read_M:0_N:0_K:1
        # CHECK-NEXT: %read_M:1_N:0_K:0
        # CHECK-NEXT: %read_M:1_N:0_K:1
        # CHECK-NEXT: %write_shared_M:0_N:0_K:0
        # CHECK-NEXT: %write_shared_M:0_N:0_K:1
        # CHECK-NEXT: %write_shared_M:1_N:0_K:0
        # CHECK-NEXT: %write_shared_M:1_N:0_K:1
        # CHECK-NEXT: %shared_memory_barrier
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1
        # CHECK-NEXT: %read_1_M:0_N:0_K:0
        # CHECK-NEXT: %read_1_M:0_N:0_K:1
        # CHECK-NEXT: %read_1_M:0_N:1_K:0
        # CHECK-NEXT: %read_1_M:0_N:1_K:1
        # CHECK-NEXT: %shared_memory_barrier_1
        # CHECK-NEXT: %write_1_shared_M:0_N:0_K:0
        # CHECK-NEXT: %write_1_shared_M:0_N:0_K:1
        # CHECK-NEXT: %write_1_shared_M:0_N:1_K:0
        # CHECK-NEXT: %write_1_shared_M:0_N:1_K:1
        # CHECK-NEXT: %shared_memory_barrier_2
        # CHECK-NEXT: %read_3_shared_M:0_N:0_K:0
        # CHECK-NEXT: %read_3_shared_M:0_N:0_K:1
        # CHECK-NEXT: %read_3_shared_M:0_N:1_K:0
        # CHECK-NEXT: %read_3_shared_M:0_N:1_K:1
        # CHECK-NEXT: %mma_M:0_N:0_K:0
        # CHECK-NEXT: %mma_M:0_N:0_K:1
        # CHECK-NEXT: %mma_M:0_N:1_K:0
        # CHECK-NEXT: %mma_M:0_N:1_K:1
        # CHECK-NEXT: %mma_M:1_N:0_K:0
        # CHECK-NEXT: %mma_M:1_N:0_K:1
        # CHECK-NEXT: %mma_M:1_N:1_K:0
        # CHECK-NEXT: %mma_M:1_N:1_K:1


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
