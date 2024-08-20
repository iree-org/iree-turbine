# RUN: python %s | FileCheck %s

import logging
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel._support.indexing import IndexingContext
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel.wave.utils import run_test, print_trace

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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]

    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
        }
    ):
        graph = read_write_same_size()
        IndexingContext.current().finalize()
        expand_graph(graph, constraints)
        print_trace(graph)
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
        # CHECK-NEXT: %write_0_0
        # CHECK-SAME: (%read_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_1
        # CHECK-SAME: (%read_1_1, %c, 4, None)
        # CHECK-NEXT: %write_1_0
        # CHECK-SAME: (%read_1_0, %c, 4, None)
        # CHECK-NEXT: %write_0_1
        # CHECK-SAME: (%read_0_1, %c, 4, None)
        # CHECK-NEXT: return

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: write(register_=read_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: write(register_=read_1_1
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: write(register_=read_1_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: write(register_=read_0_1
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: output

        # CHECK: -----


@tkw.wave_trace_only()
def read_write_different_dims(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def test_read_write():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
        }
    ):
        graph = read_write_different_dims()
        IndexingContext.current().finalize()
        expand_graph(graph, constraints)
        print_trace(graph)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%read_0_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_1
        # CHECK-SAME: (%read_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_0
        # CHECK-SAME: (%read_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_0_0_1
        # CHECK-SAME: (%read_0_0_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: write(register_=read_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M, K: ARGK*BLOCK_K}
        # CHECK-NEXT: write(register_=read_1_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K + 16}
        # CHECK-NEXT: write(register_=read_1_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K}
        # CHECK-NEXT: write(register_=read_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/256 + $WG0*BLOCK_M, K: ARGK*BLOCK_K + 16}
        # CHECK-NEXT: output

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
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
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
        graph = gemm()
        IndexingContext.current().finalize()
        expand_graph(graph, constraints)
        print_trace(graph)
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
        # CHECK-SAME: (%getresult_0_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_1_0
        # CHECK-SAME: (%getresult_1_1_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_0
        # CHECK-SAME: (%getresult_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_0_1_0
        # CHECK-SAME: (%getresult_0_1_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_0_0_0, register_0_1_0, register_1_0_0, register_1_1_0], subgraph_name=region_0, implicit_captures=[a, b])
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: write(register_=getresult_1_1_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: write(register_=getresult_1_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N}
        # CHECK-NEXT: write(register_=getresult_0_1_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16}
        # CHECK-NEXT: output

        # Reduction subgraph:

        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_1_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_0_1_0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_0_1
        # CHECK-SAME: (%a, 4, None, None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_0
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_1
        # CHECK-SAME: (%b, 4, None, None)

        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0, %acc_0_0_0)
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

        # Custom format:
        # CHECK-NEXT: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=acc_1_1_0
        # CHECK-NEXT: placeholder(_name=acc_1_0_0
        # CHECK-NEXT: placeholder(_name=acc_0_1_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_0_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_0_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_1_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) + 16 : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_1_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) + 16 : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_1_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) + 16 : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_1_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) + 16 : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_0_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_0_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: output(return_vals=([mma_0_0_1, mma_1_1_1, mma_1_0_1, mma_0_1_1],))

        # CHECK-NEXT: -----


@run_test
def test_gemm_reduction_expansion_only():
    # Note: This does not implement an actual gemm computation but reuses the
    # gemm kernel to test the expansion of the reduction subgraph.
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 16,
            BLOCK_N: 32,
            BLOCK_K: 64,
        }
    ):
        graph = gemm()
        IndexingContext.current().finalize()
        expand_graph(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_0_0_0
        # CHECK-NEXT: %register_0_1_0
        # CHECK-NEXT: %reduction
        # CHECK-NEXT: %getresult_0_1_0
        # CHECK-NEXT: %getresult_0_0_0
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%getresult_0_0_0, %c, 4, None)
        # CHECK-NEXT: %write_0_1_0
        # CHECK-SAME: (%getresult_0_1_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_0_0_0, register_0_1_0]
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N})
        # CHECK-NEXT: write(register_=getresult_0_1_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16})
        # CHECK-NEXT: output(return_vals=(None,))

        # Reduction subgraph:

        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_0_1_0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_0_2
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_0_3
        # CHECK-SAME: (%a, 4, None, None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_0_2
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_0_3
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_0
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_1
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_2
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_1_3
        # CHECK-SAME: (%b, 4, None, None)

        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0, %acc_0_0_0)
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-SAME: (%read_0_0_1, %read_0_0_1, %mma_0_0_0)
        # CHECK-NEXT: %mma_0_0_2
        # CHECK-SAME: (%read_0_0_2, %read_0_0_2, %mma_0_0_1)
        # CHECK-NEXT: %mma_0_0_3
        # CHECK-SAME: (%read_0_0_3, %read_0_0_3, %mma_0_0_2)
        # CHECK-NEXT: %mma_0_1_0
        # CHECK-SAME: (%read_0_0_0, %read_0_1_0, %acc_0_1_0)
        # CHECK-NEXT: %mma_0_1_1
        # CHECK-SAME: (%read_0_0_1, %read_0_1_1, %mma_0_1_0)
        # CHECK-NEXT: %mma_0_1_2
        # CHECK-SAME: (%read_0_0_2, %read_0_1_2, %mma_0_1_1)
        # CHECK-NEXT: %mma_0_1_3
        # CHECK-SAME: (%read_0_0_3, %read_0_1_3, %mma_0_1_2)

        # CHECK-NEXT: return [mma_0_0_3, mma_0_1_3]

        # Custom format:

        # CHECK-NEXT: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=acc_0_1_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K + 16})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K + 32})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 16, K: ARGK*BLOCK_K + 48})
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_0_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_0_0_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_2 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 32 : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_2 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 32 : 4 : 1])
        # CHECK-SAME: acc=mma_0_0_1 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_3 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 48 : 4 : 1])
        # CHECK-SAME: rhs=read_0_0_3 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 48 : 4 : 1])
        # CHECK-SAME: acc=mma_0_0_2 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)]))
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_0 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) : 4 : 1])
        # CHECK-SAME: acc=acc_0_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_1 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 16 : 4 : 1])
        # CHECK-SAME: acc=mma_0_1_0 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_0_0_2 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 32 : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_2 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 32 : 4 : 1])
        # CHECK-SAME: acc=mma_0_1_1 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: mma(lhs=read_0_0_3 (index = [$T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 48 : 4 : 1])
        # CHECK-SAME: rhs=read_0_1_3 (index = [$T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, 16*$T1 + 16*$T2 + ARGK*BLOCK_K + 4*floor($T0/16) + 48 : 4 : 1])
        # CHECK-SAME: acc=mma_0_1_2 (index = [$T0*BLOCK_M/128 + 16*$T1 + 16*$T2 + $WG0*BLOCK_M + 4*floor($T0/16) : 4 : 16, $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16]))
        # CHECK-NEXT: output(return_vals=([mma_0_0_3, mma_0_1_3],))

        # CHECK-NEXT: -----


@tkw.wave_trace_only()
def py_arithmetic_different_dims(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    a_reg = a_reg + a_reg - a_reg
    a_reg = -a_reg
    tkw.write(a_reg, c, elements_per_thread=4)


@run_test
def py_arithmetic_different_dims():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 4, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
        }
    ):
        graph = py_arithmetic_different_dims()
        IndexingContext.current().finalize()
        expand_graph(graph, constraints)
        print_trace(graph)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_1_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %add_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0)
        # CHECK-NEXT: %add_1_0_0
        # CHECK-SAME: (%read_1_0_0, %read_1_0_0)
        # CHECK-NEXT: %sub_0_0_0
        # CHECK-SAME: (%add_0_0_0, %read_0_0_0)
        # CHECK-NEXT: %sub_1_0_0
        # CHECK-SAME: (%add_1_0_0, %read_1_0_0)
        # CHECK-NEXT: %neg_0_0_0
        # CHECK-SAME: (%sub_0_0_0,)
        # CHECK-NEXT: %neg_1_0_0
        # CHECK-SAME: (%sub_1_0_0,)
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%neg_0_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_1
        # CHECK-SAME: (%neg_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_1_0_0
        # CHECK-SAME: (%neg_1_0_0, %c, 4, None)
        # CHECK-NEXT: %write_0_0_1
        # CHECK-SAME: (%neg_0_0_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: add(lhs=read_0_0_0, rhs=read_0_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: add(lhs=read_1_0_0, rhs=read_1_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: sub(lhs=add_0_0_0, rhs=read_0_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: sub(lhs=add_1_0_0, rhs=read_1_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: neg(arg=sub_0_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: neg(arg=sub_1_0_0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + $WG1*BLOCK_N})
        # CHECK-NEXT: write(register_=neg_0_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: $WG2*BLOCK_K})
        # CHECK-NEXT: write(register_=neg_1_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, K: $WG2*BLOCK_K + 16})
        # CHECK-NEXT: write(register_=neg_1_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 16, K: $WG2*BLOCK_K})
        # CHECK-NEXT: write(register_=neg_0_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M, K: $WG2*BLOCK_K + 16})

        # CHECK: -----


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
