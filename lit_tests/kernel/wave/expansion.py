# RUN: python %s | FileCheck %s

import logging
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion import expand_graph
from iree.turbine.kernel.wave.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import run_test, print_trace
from iree.turbine.kernel.wave.constraints import MMAType
import sympy

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B

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
    constraints += [tkw.WaveConstraint(M, BLOCK_M, sympy.floor(THREAD_0 / 64))]
    constraints += [tkw.WaveConstraint(N, BLOCK_N, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 16, N: 16},
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
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
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
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64), N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64), N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_0_0
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64), N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_1_1
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_1_0
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_0_1
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64), N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
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
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N, THREAD_1)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K, THREAD_2)]
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
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
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
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=read_1_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_1_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=read_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
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
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
        }
    ):
        graph = gemm()
        IndexingContext.current().finalize()
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
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
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_0_0_0, register_0_1_0, register_1_0_0, register_1_1_0], subgraph_name=region_0, implicit_captures=[a, b])
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}
        # CHECK-NEXT: write(register_=getresult_1_1_0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}
        # CHECK-NEXT: write(register_=getresult_1_0_0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}
        # CHECK-NEXT: write(register_=getresult_0_1_0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}
        # CHECK-NEXT: output

        # Reduction subgraph:

        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_0_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_1_1_0

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
        # CHECK-NEXT: return [mma_0_0_1, mma_0_1_1, mma_1_0_1, mma_1_1_1]

        # Custom format:
        # CHECK-NEXT: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=acc_0_1_0
        # CHECK-NEXT: placeholder(_name=acc_1_0_0
        # CHECK-NEXT: placeholder(_name=acc_1_1_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_1_1_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_1_1_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_1_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_1_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_0_1_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_0_1_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: output(return_vals=([mma_0_0_1, mma_0_1_1, mma_1_0_1, mma_1_1_1],))

        # CHECK-NEXT: -----


@tkw.wave_trace_only()
def batched_gemm(
    a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[B, M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[B, M, N, tkl.f32]) -> tkl.Register[B, M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_batched_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    # Since the MMA shapes only cover M, N and K, we specify the canonical shape for
    # the batch dimension in the vector_shapes.
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            vector_shapes={B: 0},
            mma_type=MMAType.F32_16x16x16_F16,
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            BLOCK_B: 1,
        }
    ):
        graph = batched_gemm()
        IndexingContext.current().finalize()
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
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
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_0_0_0, register_0_1_0, register_1_0_0, register_1_1_0], subgraph_name=region_0, implicit_captures=[a, b])
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0
        # CHECK-SAME: index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}
        # CHECK-NEXT: write(register_=getresult_1_1_0
        # CHECK-SAME: index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}
        # CHECK-NEXT: write(register_=getresult_1_0_0
        # CHECK-SAME: index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}
        # CHECK-NEXT: write(register_=getresult_0_1_0
        # CHECK-SAME: index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}
        # CHECK-NEXT: output

        # Reduction subgraph:

        # CHECK: %acc_0_0_0
        # CHECK-NEXT: %acc_0_1_0
        # CHECK-NEXT: %acc_1_0_0
        # CHECK-NEXT: %acc_1_1_0

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
        # CHECK-NEXT: return [mma_0_0_1, mma_0_1_1, mma_1_0_1, mma_1_1_1]

        # Custom format:
        # CHECK-NEXT: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=acc_0_1_0
        # CHECK-NEXT: placeholder(_name=acc_1_0_0
        # CHECK-NEXT: placeholder(_name=acc_1_1_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_0 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_0_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_1 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_0_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_0 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_1_1_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_1 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_1_1_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_1_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_0 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_1_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_1_0_1 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_1 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_1_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_0 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_0_1_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_1_1 (index = {B: $WG2*BLOCK_B, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_0_1_0 (index = {B: $WG2*BLOCK_B, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16}))
        # CHECK-NEXT: output(return_vals=([mma_0_0_1, mma_0_1_1, mma_1_0_1, mma_1_1_1],))

        # CHECK-NEXT: -----


@tkw.wave_trace_only()
def gemm_non_direct_acc(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        new_acc = tkw.exp2(a_reg) + acc
        acc = tkw.mma(a_reg, b_reg, new_acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_gemm_non_direct_acc():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
        }
    ):
        graph = gemm_non_direct_acc()
        IndexingContext.current().finalize()
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: %add_0_0_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.add](args = (%exp2_0_0_0, %acc_0_0_0), kwargs = {})
        # CHECK: %add_1_1_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.add](args = (%exp2_1_0_0, %acc_1_1_0), kwargs = {})
        # CHECK: %add_1_0_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.add](args = (%exp2_1_0_0, %acc_1_0_0), kwargs = {})
        # CHECK: %add_0_1_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.add](args = (%exp2_0_0_0, %acc_0_1_0), kwargs = {})
        # CHECK: %mma_0_0_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_0_0_0, %read_0_0_0, %add_0_0_0), kwargs = {})
        # CHECK: %mma_0_0_1
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_0_0_1, %read_0_0_1, %mma_0_0_0), kwargs = {})
        # CHECK: %mma_1_1_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_1_0_0, %read_0_1_0, %add_1_1_0), kwargs = {})
        # CHECK: %mma_1_1_1
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_1_0_1, %read_0_1_1, %mma_1_1_0), kwargs = {})
        # CHECK: %mma_1_0_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_1_0_0, %read_0_0_0, %add_1_0_0), kwargs = {})
        # CHECK: %mma_1_0_1
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_1_0_1, %read_0_0_1, %mma_1_0_0), kwargs = {})
        # CHECK: %mma_0_1_0
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_0_0_0, %read_0_1_0, %add_0_1_0), kwargs = {})
        # CHECK: %mma_0_1_1
        # CHECK-SAME: call_function[target=iree.turbine.kernel.ops.wave_ops.mma](args = (%read_0_0_1, %read_0_1_1, %mma_0_1_0), kwargs = {})


@tkw.wave_trace_only()
def tiled_max(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
):
    init_max = tkl.Register[M, tkl.f16](-1e6)

    @tkw.reduction(K, init_args=[init_max])
    def repeat(acc: tkl.Register[M, tkl.f16]) -> tkl.Register[M, tkl.f16]:
        a_reg = tkw.read(a, elements_per_thread=4)
        partial_max = tkw.max(a_reg, acc, dim=K)
        return partial_max

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_tiled_max():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 1, 1),
            vector_shapes={M: 16, K: 4},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 64,
            BLOCK_K: 32,
        }
    ):
        graph = tiled_max()
        IndexingContext.current().finalize()
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: max(arg=[read_0_0, read_0_1, read_0_2, read_0_3, read_0_4, read_0_5, read_0_6, read_0_7], init=acc_0_0
        # CHECK: max(arg=[read_1_0, read_1_1, read_1_2, read_1_3, read_1_4, read_1_5, read_1_6, read_1_7], init=acc_1_0
        # CHECK: output(return_vals=([max_0_0, max_1_0],))
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
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
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
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_0_0_0
        # CHECK-NEXT: %reduction
        # CHECK-NEXT: %getresult_0_0_0
        # CHECK-NEXT: %write_0_0_0
        # CHECK-SAME: (%getresult_0_0_0, %c, 4, None)
        # CHECK-NEXT: return None

        # Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_0_0_0]
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=getresult_0_0_0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)})
        # CHECK-NEXT: output(return_vals=(None,))

        # Reduction subgraph:

        # CHECK: %acc_0_0_0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%a, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%a, 4, None, None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_0_0_0
        # CHECK-SAME: (%b, 4, None, None)
        # CHECK-NEXT: %read_0_0_1
        # CHECK-SAME: (%b, 4, None, None)

        # CHECK-NEXT: %mma_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0, %acc_0_0_0)
        # CHECK-NEXT: %mma_0_0_1
        # CHECK-SAME: (%read_0_0_1, %read_0_0_1, %mma_0_0_0)

        # CHECK-NEXT: return [mma_0_0_1]

        # Custom format:

        # CHECK-NEXT: placeholder(_name=acc_0_0_0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: mma(lhs=read_0_0_1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_0_0_1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16), K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_0_0_0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16)}))
        # CHECK-NEXT: output(return_vals=([mma_0_0_1],))

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
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
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
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: add(lhs=read_0_0_0, rhs=read_0_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: add(lhs=read_1_0_0, rhs=read_1_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: sub(lhs=add_0_0_0, rhs=read_0_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: sub(lhs=add_1_0_0, rhs=read_1_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: neg(arg=sub_0_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: neg(arg=sub_1_0_0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=neg_0_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, K: 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=neg_1_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, K: 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=neg_1_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16, K: 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=neg_0_0_0, memory=c, elements_per_thread=4, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M, K: 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}

        # CHECK: -----


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
