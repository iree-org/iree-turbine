# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.promotion import promote_node, promote_placeholders
from iree.turbine.kernel.wave.hoisting import hoist_loop_invariant_ops
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.ops.wave_ops import *
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.utils.print_utils import print_trace


def get_read_nodes(graph: fx.Graph) -> list[CustomOp]:
    custom_nodes: list[CustomOp] = [get_custom(node) for node in graph.nodes]
    return [node for node in custom_nodes if isinstance(node, Read)]


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
        tkw.initialize_and_check_constraints(constraints, IndexingContext.current())
        infer_types(trace)
        promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        print_trace(trace, False)
        # CHECK: %allocate
        # CHECK-SAME: ((M, N), (BLOCK_M, BLOCK_N + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %write_1
        # CHECK-SAME: (%read, %allocate, 4, None, ())
        # CHECK-NEXT: %read_1
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1])
        # CHECK-NEXT: %write
        # CHECK-SAME: (%read_1, %c, 4, None, ())

        # CHECK: -----


@tkw.wave_trace_only()
def read_write_same_size_different_address_spaces(
    a: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE_1, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def test_read_write_equal_sizes_different_address_spaces():
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
            ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_1: GLOBAL_ADDRESS_SPACE,
        }
    ):
        trace: CapturedTrace = read_write_same_size_different_address_spaces()
        IndexingContext.current().finalize()
        tkw.initialize_and_check_constraints(constraints, IndexingContext.current())
        infer_types(trace)
        promote_placeholders(trace, constraints)
        print_trace(trace, False)
        # CHECK: %allocate
        # CHECK-SAME: ((M, N), (BLOCK_M, BLOCK_N + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %write_1
        # CHECK-SAME: (%read, %allocate, 4, None, ())
        # CHECK-NEXT: %read_1
        # CHECK-SAME: (%allocate, 4, None, (), [%write_1])
        # CHECK-NEXT: %write
        # CHECK-SAME: (%read_1, %c, 4, None, ())

        # CHECK: -----


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.iterate(K, init_args=[c_reg])
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
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]
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
        read_nodes = get_read_nodes(graph)
        IndexingContext.current().finalize()
        tkw.initialize_and_check_constraints(constraints, IndexingContext.current())
        infer_types(trace)
        for read_node in read_nodes:
            promote_node(read_node, None, SHARED_ADDRESS_SPACE, constraints)
        hoist_loop_invariant_ops(trace, constraints)
        print_trace(trace, False)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register
        # CHECK: %allocate_1
        # CHECK-SAME: ((N, K), (BLOCK_N, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: %allocate
        # CHECK-SAME: ((M, K), (BLOCK_M, BLOCK_K + 4), f16, $SHARED_ADDRESS_SPACE, 4)
        # CHECK-NEXT: iterate
        # CHECK-NEXT: %write
        # CHECK-SAME: (%iterate, %c, 4, None, ())

        # iterate subgraph:
        # CHECK: %b
        # CHECK-NEXT: %a
        # CHECK-NEXT: %acc
        # CHECK-NEXT: %read
        # CHECK-NEXT: %write
        # CHECK-SAME: (%read, %allocate, 4, None, ())
        # CHECK-NEXT: %read_2
        # CHECK-SAME: (%allocate, 4, None, (), [%write])
        # CHECK-NEXT: %read_1
        # CHECK-NEXT: %write_1
        # CHECK-SAME: (%read_1, %allocate_1, 4, None, ())
        # CHECK-NEXT: %read_3
        # CHECK-SAME: (%allocate_1, 4, None, (), [%write_1])
        # CHECK-NEXT: %mma
        # CHECK-SAME: (%read_2, %read_3, %acc, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
