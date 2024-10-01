# RUN: python %s | FileCheck %s

import logging
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.promotion import promote_placeholders
from shark_turbine.kernel.wave.hoisting import hoist_allocs
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel._support.indexing import IndexingContext
from shark_turbine.kernel.ops.wave_ops import *
from shark_turbine.kernel.wave.utils import run_test, print_subgraph
from shark_turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from shark_turbine.kernel.wave.shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
)
from shark_turbine.kernel.wave.scheduling.schedule import schedule_graph


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Address space
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm_pipelined(
    a: tkl.Memory[M, K, ADDRESS_SPACE_0, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE_0, tkl.f16],
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
def test_gemm_pipelined():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 256,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
            ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            SHARED_MEMORY_UNITS: 2,
            GLOBAL_MEMORY_UNITS: 2,
            MMA_UNITS: 2,
        }
    ):
        trace: CapturedTrace = gemm_pipelined()
        IndexingContext.current().finalize()
        promote_placeholders(trace, constraints)
        hoist_allocs(trace)
        expand_graph(trace, constraints)
        minimize_global_loads(trace, constraints)
        apply_shared_memory_indexing_corrections(trace, constraints)
        schedule_graph(trace, constraints)

        print_subgraph(trace, "pipelined_reduction", False)
        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_0_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_1_1_0
        # CHECK-NEXT: %rotating_reg_0
        # CHECK-NEXT: %rotating_reg_1
        # CHECK-NEXT: %rotating_reg_2
        # CHECK-NEXT: %rotating_reg_3
        # CHECK-NEXT: %rotating_reg_4
        # CHECK-NEXT: %rotating_reg_5
        # CHECK-NEXT: %rotating_reg_6
        # CHECK-NEXT: %mma_1_1_1
        # CHECK-SAME: (%rotating_reg_1, %rotating_reg_4, %rotating_reg_6)
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_4
        # CHECK-NEXT: %read_5
        # CHECK-NEXT: %read_shared_1_0_0
        # CHECK-NEXT: %read_shared_1_0_1
        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_shared_0_0_0, %read_shared_0_0_1, %acc_0_0_0)
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-SAME: (%read_shared_0_0_0, %rotating_reg_3, %acc_0_1_0)
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-SAME: (%rotating_reg_0, %rotating_reg_2, %mma_0_0_0)
        # CHECK-NEXT: %mma_1_0_0
        # CHECK-SAME: (%read_shared_1_0_0, %read_shared_0_0_1, %acc_1_0_0)
        # CHECK-NEXT: %write_2
        # CHECK-NEXT: %write_3
        # CHECK-NEXT: %mma_1_0_1
        # CHECK-SAME: (%read_shared_1_0_1, %rotating_reg_2, %mma_1_0_0)
        # CHECK-NEXT: %mma_0_1_1
        # CHECK-SAME: (%rotating_reg_0, %rotating_reg_5, %mma_0_1_0)
        # CHECK-NEXT: %read_shared_0_1_0
        # CHECK-NEXT: %read_shared_0_1_1
        # CHECK-NEXT: %mma_1_1_0
        # CHECK-SAME: (%read_shared_1_0_0, %rotating_reg_3, %mma_1_1_1)
        # CHECK-NEXT: %read_shared_0_0_2
        # CHECK-NEXT: %read_shared_0_0_3
        # CHECK-NEXT: [mma_0_0_1, mma_0_1_1, mma_1_0_1, mma_1_1_1, read_shared_0_0_2, read_shared_1_0_1, read_shared_0_0_3, read_shared_0_1_0, rotating_reg_5, read_shared_0_1_1, mma_1_1_0]

        print_subgraph(trace, "region_1", False)
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_0_0_0
        # CHECK-NEXT: %register_1_1_0
        # CHECK-NEXT: %register_1_0_0
        # CHECK-NEXT: %register_0_1_0
        # CHECK-NEXT: %allocate
        # CHECK-NEXT: %allocate_1
        # CHECK-NEXT: %read_4
        # CHECK-NEXT: %read_5
        # CHECK-NEXT: %write_2
        # CHECK-NEXT: %write_3
        # CHECK-NEXT: %read_shared_0_1_0
        # CHECK-NEXT: %read_shared_0_1_1
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_shared_0_0_2
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_3
        # CHECK-NEXT: %read_6
        # CHECK-NEXT: %read_7
        # CHECK-NEXT: %read_shared_1_0_0
        # CHECK-NEXT: %read_shared_1_0_1
        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_shared_0_0_0, %read_shared_0_0_3, %register_0_0_0)
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-SAME: (%read_shared_0_0_0, %read_shared_0_1_0, %register_0_1_0)
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-SAME: (%read_shared_0_0_1, %read_shared_0_0_2, %mma_0_0_0)
        # CHECK-NEXT: %mma_1_0_0
        # CHECK-SAME: (%read_shared_1_0_0, %read_shared_0_0_3, %register_1_0_0)
        # CHECK-NEXT: %write_4
        # CHECK-NEXT: %write_5
        # CHECK-NEXT: %mma_1_0_1
        # CHECK-SAME: (%read_shared_1_0_1, %read_shared_0_0_2, %mma_1_0_0)
        # CHECK-NEXT: %mma_0_1_1
        # CHECK-SAME: (%read_shared_0_0_1, %read_shared_0_1_1, %mma_0_1_0)
        # CHECK-NEXT: %read_shared_0_1_2
        # CHECK-NEXT: %read_shared_0_1_3
        # CHECK-NEXT: %mma_1_1_0
        # CHECK-SAME: (%read_shared_1_0_0, %read_shared_0_1_0, %register_1_1_0)
        # CHECK-NEXT: %read_shared_0_0_4
        # CHECK-NEXT: %read_shared_0_0_5
        # CHECK-NEXT: %reduction_1
        # CHECK-NEXT: %getresult_1_1_0
        # CHECK-NEXT: %getresult_1_0_0
        # CHECK-NEXT: %getresult_0_1_0
        # CHECK-NEXT: %getresult_0_0_0
        # CHECK-NEXT: %get_result_4
        # CHECK-NEXT: %get_result_5
        # CHECK-NEXT: %get_result_6
        # CHECK-NEXT: %get_result_7
        # CHECK-NEXT: %get_result_8
        # CHECK-NEXT: %get_result_9
        # CHECK-NEXT: %get_result_10
        # CHECK-NEXT: %mma_1_1_1
        # CHECK-SAME: (%get_result_5, %get_result_9, %get_result_10)
        # CHECK-NEXT: %read_shared_0_0_6
        # CHECK-NEXT: %read_shared_0_0_7
        # CHECK-NEXT: %read_shared_1_0_2
        # CHECK-NEXT: %read_shared_1_0_3
        # CHECK-NEXT: %mma_0_0_2
        # CHECK-SAME: (%read_shared_0_0_6, %read_shared_0_0_7, %getresult_0_0_0)
        # CHECK-NEXT: %mma_0_1_2
        # CHECK-SAME: (%read_shared_0_0_6, %get_result_7, %getresult_0_1_0)
        # CHECK-NEXT: %mma_0_0_3
        # CHECK-SAME: (%get_result_4, %get_result_6, %mma_0_0_2)
        # CHECK-NEXT: %mma_1_0_2
        # CHECK-SAME: (%read_shared_1_0_2, %read_shared_0_0_7, %getresult_1_0_0)
        # CHECK-NEXT: %mma_1_0_3
        # CHECK-SAME: (%read_shared_1_0_3, %get_result_6, %mma_1_0_2)
        # CHECK-NEXT: %mma_0_1_3
        # CHECK-SAME: (%get_result_4, %get_result_9, %mma_0_1_2)
        # CHECK-NEXT: %mma_1_1_2
        # CHECK-SAME: (%read_shared_1_0_2, %get_result_7, %mma_1_1_1)
        # CHECK-NEXT: %mma_1_1_3
        # CHECK-SAME: (%read_shared_1_0_3, %get_result_9, %mma_1_1_2)
        # CHECK-NEXT: %write_0_0_0
        # CHECK-NEXT: %write_1_1_0
        # CHECK-NEXT: %write_1_0_0
        # CHECK-NEXT: %write_0_1_0
        # CHECK-NEXT: return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
