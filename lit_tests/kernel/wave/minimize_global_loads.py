# RUN: python %s | FileCheck %s

import logging
from typing import Callable
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.promotion import promote_placeholders
from shark_turbine.kernel.wave.hoisting import hoist_allocs
from shark_turbine.kernel.wave.barriers import add_shared_memory_barriers
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel._support.indexing import IndexingContext
from shark_turbine.kernel.ops.wave_ops import *
from shark_turbine.kernel.wave.utils import run_test, print_trace
from shark_turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from shark_turbine.kernel.wave.visualization import visualize_graph


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

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm(
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
def test_gemm():
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
            K: 64,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 64,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
            ADDRESS_SPACE_0: SHARED_ADDRESS_SPACE,
        }
    ):
        trace: CapturedTrace = gemm()
        visualize = False
        IndexingContext.current().finalize()
        promote_placeholders(trace, constraints)
        hoist_allocs(trace)
        expand_graph(trace, constraints)
        if visualize:
            visualize_graph(trace.get_subgraph("region_0"), "before.png")
        minimize_global_loads(trace, constraints)
        if visualize:
            visualize_graph(trace.get_subgraph("region_0"), "after.png")
        add_shared_memory_barriers(trace)
        print_trace(trace)
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

        # Root graph (custom format):
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register
        # CHECK-SAME: index={M: $WG0*BLOCK_M, N: $WG1*BLOCK_N + BLOCK_N/2})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 16, N: $WG1*BLOCK_N + BLOCK_N/2 + 16})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 16, N: $WG1*BLOCK_N + BLOCK_N/2})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M, N: $WG1*BLOCK_N + BLOCK_N/2 + 16})
        # CHECK-NEXT: allocate(
        # CHECK-NEXT: allocate(
        # CHECK-NEXT: reduction(
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0, memory=c
        # CHECK-SAME: index={M: $WG0*BLOCK_M, N: $WG1*BLOCK_N + BLOCK_N/2})
        # CHECK-NEXT: write(register_=getresult_1_1_0, memory=c
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 16, N: $WG1*BLOCK_N + BLOCK_N/2 + 16})
        # CHECK-NEXT: write(register_=getresult_1_0_0, memory=c
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 16, N: $WG1*BLOCK_N + BLOCK_N/2})
        # CHECK-NEXT: write(register_=getresult_0_1_0, memory=c
        # CHECK-SAME: index={M: $WG0*BLOCK_M, N: $WG1*BLOCK_N + BLOCK_N/2 + 16})

        # Reduction subgraph:
        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_1_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_0_1_0
        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_4
        # CHECK-SAME: (%a, 8, None, None)
        # CHECK-NEXT: %write_2
        # CHECK-SAME: (%read_4, %allocate, 8, None)
        # CHECK-NEXT: %read_5
        # CHECK-SAME: (%a, 8, None, None)
        # CHECK-NEXT: %write_3
        # CHECK-SAME: (%read_5, %allocate, 8, None)
        # CHECK-NEXT: %barrier
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_shared_0_0_2
        # CHECK-NEXT: %read_shared_0_0_3
        # CHECK-NEXT: %read_shared_1_0_0
        # CHECK-NEXT: %read_shared_1_0_1
        # CHECK-NEXT: %read_shared_1_0_2
        # CHECK-NEXT: %read_shared_1_0_3
        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_6
        # CHECK-SAME: (%b, 8, None, None)
        # CHECK-NEXT: %write_4
        # CHECK-SAME: (%read_6, %allocate_1, 8, None)
        # CHECK-NEXT: %read_7
        # CHECK-SAME: (%b, 8, None, None)
        # CHECK-NEXT: %write_5
        # CHECK-SAME: (%read_7, %allocate_1, 8, None)
        # CHECK-NEXT: %barrier_1
        # CHECK-NEXT: %read_shared_0_0_0
        # CHECK-NEXT: %read_shared_0_0_1
        # CHECK-NEXT: %read_shared_0_0_2
        # CHECK-NEXT: %read_shared_0_0_3
        # CHECK-NEXT: %read_shared_0_1_0
        # CHECK-NEXT: %read_shared_0_1_1
        # CHECK-NEXT: %read_shared_0_1_2
        # CHECK-NEXT: %read_shared_0_1_3
        # CHECK-NEXT: %mma_0_0_0
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-NEXT: %mma_0_0_2
        # CHECK-NEXT: %mma_0_0_3
        # CHECK-NEXT: %mma_1_1_0
        # CHECK-NEXT: %mma_1_1_1
        # CHECK-NEXT: %mma_1_1_2
        # CHECK-NEXT: %mma_1_1_3
        # CHECK-NEXT: %mma_1_0_0
        # CHECK-NEXT: %mma_1_0_1
        # CHECK-NEXT: %mma_1_0_2
        # CHECK-NEXT: %mma_1_0_3
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-NEXT: %mma_0_1_1
        # CHECK-NEXT: %mma_0_1_2
        # CHECK-NEXT: %mma_0_1_3

        # Reduction subgraph (custom format):
        # CHECK: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=acc_1_1_0
        # CHECK-NEXT: placeholder(_name=acc_1_0_0
        # CHECK-NEXT: placeholder(_name=acc_0_1_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=8,
        # CHECK-SAME: index={M: $WG0*BLOCK_M + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 8 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_4, memory=allocate, elements_per_thread=8,
        # CHECK-SAME: index={M: Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 8 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=8,
        # CHECK-SAME: index={M: $WG0*BLOCK_M + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 8 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_5, memory=allocate, elements_per_thread=8,
        # CHECK-SAME: index={M: Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 8 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: barrier()
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, _write_dependency=[write_2, write_3], index={M: $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: placeholder(_name=b, _type=Memory[N, K].of(f16))
        # CHECK-NEXT: read(memory=b, elements_per_thread=8,
        # CHECK-SAME: index={N: $WG1*BLOCK_N + BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 8 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_6, memory=allocate_1, elements_per_thread=8,
        # CHECK-SAME: index={N: BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 8 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=8,
        # CHECK-SAME: index={N: $WG1*BLOCK_N + BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 8 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_7, memory=allocate_1, elements_per_thread=8,
        # CHECK-SMAE: index={N: BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 8 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: barrier()
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2 + 16, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2 + 16, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2 + 16, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, _write_dependency=[write_4, write_5], index={N: $WG1*BLOCK_N + BLOCK_N/2 + 16, K: ARGK*BLOCK_K + 48})


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
