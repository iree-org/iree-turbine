# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.print_utils import (
    print_trace,
)
from iree.turbine.kernel.wave.utils.graph_utils import (
    initialize_iter_args,
)
from iree.turbine.kernel.wave.constraints import MMAType
import sympy

# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
K1 = tkl.sym.K1
K2 = tkl.sym.K2

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
BLOCK_K2 = tkl.sym.BLOCK_K2

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
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_M:0_N:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:0_N:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %write_M:0_N:0
        # CHECK-SAME: (%read_M:0_N:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:1
        # CHECK-SAME: (%read_M:0_N:1, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0
        # CHECK-SAME: (%read_M:1_N:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:1
        # CHECK-SAME: (%read_M:1_N:1, %c, 4, None, ())
        # CHECK-NEXT: return

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16 : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16 : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:0_N:0
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:0_N:1
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:1_N:0
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16 : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:1_N:1
        # CHECK-SAME: index={M: $T0 + $WG0*BLOCK_M + BLOCK_M*floor($T0/64) + 16 : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N + 16 : 4 : 1}
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
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:0_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:1
        # CHECK-SAME: (%read_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, N: $T1*BLOCK_N + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:0_N:0_K:0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M : 1 : 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:0_N:0_K:0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M : 1 : 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:1_N:0_K:0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=read_M:1_N:0_K:0
        # CHECK-SAME: index={M: $T0*BLOCK_M/64 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, K: $T2*BLOCK_K + 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
        # CHECK-NEXT: output

        # CHECK: -----


@tkw.wave_trace_only()
def write_in_reduction(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
):
    # TODO(#364): simplify this by removing the max once it's possible to have a
    # loop/reduction without reducing over something.
    init_max = tkl.Register[M, tkl.f16](-1e6)

    # TODO: Cannot deduce elements_per_thread for a_reg without a workgroup constraint yet.
    @tkw.reduction(K, init_args=[init_max])
    def repeat(acc: tkl.Register[M, tkl.f16]) -> tkl.Register[M, tkl.f16]:
        a_reg = tkw.read(a, elements_per_thread=4)
        tkw.write(a_reg, b, elements_per_thread=4)
        return a_reg

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_write_in_reduction():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 1, 1),
            vector_shapes={M: 16, K: 16},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_K: 32,
        }
    ):
        graph = write_in_reduction()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: graph()
        # CHECK: %reduction :
        # CHECK-SAME: args = (K,
        # CHECK: %get_result_M:0_K:0 :
        # CHECK-SAME: args = (%reduction, 0)
        # CHECK: %write_M:0_K:0 :
        # CHECK-SAME: (%get_result_M:0_K:0, %c, 4,

        # CHECK: Custom format:
        # CHECK: reduction(axis=K,
        # CHECK: get_result(value=reduction, res_idx=0)
        # CHECK: write(register_=get_result_M:0_K:0, memory=c, elements_per_thread=4,

        # Reduction subgraph:
        # CHECK: graph():
        # CHECK: %read_M:0_K:0 :
        # CHECK-SAME: (args = (%a, 4,
        # CHECK: %read_M:0_K:1 :
        # CHECK-SAME: (args = (%a, 4,
        # CHECK: %write_M:0_K:0 :
        # CHECK-SAME: (%read_M:0_K:0, %b, 4,
        # CHECK: %write_M:0_K:1 :
        # CHECK-SAME: (%read_M:0_K:1, %b, 4,

        # CHECK: Custom format:
        # CHECK: read(memory=a, elements_per_thread=4,
        # CHECK: read(memory=a, elements_per_thread=4,
        # CHECK: write(register_=read_M:0_K:0, memory=b, elements_per_thread=4,
        # CHECK: write(register_=read_M:0_K:1, memory=b, elements_per_thread=4,
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


@tkw.wave_trace_only()
def no_writes(a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16]):
    tkw.read(a, elements_per_thread=16)


@run_test
def test_no_writes():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 1, 1),
            vector_shapes={M: 16, K: 16},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_K: 32,
        }
    ):
        graph = no_writes()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)

        # CHECK: graph():
        # CHECK: %a :
        # CHECK-SAME: [num_users=1] = placeholder[target=a]
        # CHECK: %read :
        # CHECK-SAME: (args = (%a, 16, None, (), None), kwargs = {})
        # CHECK: return None
        # CHECK: Custom format:
        # CHECK: placeholder(_name=a, _type=Memory[M, K].of(f16))
        # CHECK: read(memory=a, elements_per_thread=16
        # CHECK: output(return_vals=(None,))

        # CHECK: -----


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
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: region_1 [root]:
        # CHECK: graph():
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_M:0_N:0_K:0
        # CHECK-NEXT: %register_M:0_N:1_K:0
        # CHECK-NEXT: %register_M:1_N:0_K:0
        # CHECK-NEXT: %register_M:1_N:1_K:0
        # CHECK-NEXT: %reduction
        # CHECK-SAME: %register_M:0_N:0_K:0, %register_M:0_N:1_K:0, %register_M:1_N:0_K:0, %register_M:1_N:1_K:0
        # CHECK-NEXT: %get_result_M:0_N:0_K:0
        # CHECK-NEXT: %get_result_M:0_N:1_K:0
        # CHECK-NEXT: %get_result_M:1_N:0_K:0
        # CHECK-NEXT: %get_result_M:1_N:1_K:0
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:1_K:0
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:0
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:1_K:0
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_M:0_N:0_K:0, register_M:0_N:1_K:0, register_M:1_N:0_K:0, register_M:1_N:1_K:0], subgraph_name=region_0, implicit_captures=[a, b])
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: write(register_=get_result_M:0_N:0_K:0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:0_N:1_K:0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:1_N:0_K:0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:1_N:1_K:0
        # CHECK-SAME: index={M:  $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}
        # CHECK-NEXT: output

        # Reduction subgraph:
        # CHECK: region_0:
        # CHECK: graph():
        # CHECK: %acc_M:0_N:0_K:0
        # CHECK-NEXT: %acc_M:0_N:1_K:0
        # CHECK-NEXT: %acc_M:1_N:0_K:0
        # CHECK-NEXT: %acc_M:1_N:1_K:0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:0_N:0_K:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:1
        # CHECK-SAME: (%a, 4, None, (), None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_1_M:0_N:0_K:0
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:0_K:1
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:1_K:0
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:1_K:1
        # CHECK-SAME: (%b, 4, None, (), None)

        # CHECK-NEXT: %mma_M:0_N:0_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_1_M:0_N:0_K:0, %acc_M:0_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:0_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:0_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:1_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_1_M:0_N:1_K:0, %acc_M:0_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:1_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:0_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:0_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %read_1_M:0_N:0_K:0, %acc_M:1_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:0_K:1
        # CHECK-SAME: (%read_M:1_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:1_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:1_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %read_1_M:0_N:1_K:0, %acc_M:1_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:1_K:1
        # CHECK-SAME: (%read_M:1_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:1_N:1_K:0, None)
        # CHECK-NEXT: return [mma_M:0_N:0_K:1, mma_M:0_N:1_K:1, mma_M:1_N:0_K:1, mma_M:1_N:1_K:1]

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:1_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:1_K:0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:0_N:1_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:0_N:1_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:1_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:1_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:1_N:1_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:1_N:1_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: output(return_vals=([mma_M:0_N:0_K:1, mma_M:0_N:1_K:1, mma_M:1_N:0_K:1, mma_M:1_N:1_K:1],))

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
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_M:0_N:0_K:0
        # CHECK-NEXT: %register_M:0_N:1_K:0
        # CHECK-NEXT: %register_M:1_N:0_K:0
        # CHECK-NEXT: %register_M:1_N:1_K:0
        # CHECK-NEXT: %reduction
        # CHECK-SAME: %register_M:0_N:0_K:0, %register_M:0_N:1_K:0, %register_M:1_N:0_K:0, %register_M:1_N:1_K:0
        # CHECK-NEXT: %get_result_M:0_N:0_K:0
        # CHECK-NEXT: %get_result_M:0_N:1_K:0
        # CHECK-NEXT: %get_result_M:1_N:0_K:0
        # CHECK-NEXT: %get_result_M:1_N:1_K:0
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:1_K:0
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:0
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:1_K:0
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(shape=(B, M, N), dtype=f32, value=0.0, index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_M:0_N:0_K:0, register_M:0_N:1_K:0, register_M:1_N:0_K:0, register_M:1_N:1_K:0], subgraph_name=region_0, implicit_captures=[a, b])
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: write(register_=get_result_M:0_N:0_K:0
        # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:0_N:1_K:0
        # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:1_N:0_K:0
        # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}
        # CHECK-NEXT: write(register_=get_result_M:1_N:1_K:0
        # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}
        # CHECK-NEXT: output

        # Reduction subgraph:

        # CHECK: %acc_M:0_N:0_K:0
        # CHECK-NEXT: %acc_M:0_N:1_K:0
        # CHECK-NEXT: %acc_M:1_N:0_K:0
        # CHECK-NEXT: %acc_M:1_N:1_K:0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:0_N:0_K:1
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:1
        # CHECK-SAME: (%a, 4, None, (), None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_1_M:0_N:0_K:0
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:0_K:1
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:1_K:0
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:1_K:1
        # CHECK-SAME: (%b, 4, None, (), None)

        # CHECK-NEXT: %mma_M:0_N:0_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_1_M:0_N:0_K:0, %acc_M:0_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:0_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:0_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:1_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_1_M:0_N:1_K:0, %acc_M:0_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:1_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:0_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:0_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %read_1_M:0_N:0_K:0, %acc_M:1_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:0_K:1
        # CHECK-SAME: (%read_M:1_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:1_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:1_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %read_1_M:0_N:1_K:0, %acc_M:1_N:1_K:0, None)
        # CHECK-NEXT: %mma_M:1_N:1_K:1
        # CHECK-SAME: (%read_M:1_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:1_N:1_K:0, None)
        # CHECK-NEXT: return [mma_M:0_N:0_K:1, mma_M:0_N:1_K:1, mma_M:1_N:0_K:1, mma_M:1_N:1_K:1]

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:1_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:1_K:0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:0 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:0     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:0_N:0_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:1 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:1     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:0_N:0_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:0 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:0     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:0_N:1_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:1 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:1     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:0_N:1_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:0 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:0     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:1_N:0_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:1 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:1     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:1_N:0_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:0 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:0     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:1_N:1_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:1_N:0_K:1 (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:1_K:1     (index = {B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:1_N:1_K:0      (index = {B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1}))
        # CHECK-NEXT: output(return_vals=([mma_M:0_N:0_K:1, mma_M:0_N:1_K:1, mma_M:1_N:0_K:1, mma_M:1_N:1_K:1],))

        # CHECK-NEXT: -----


@tkw.wave_trace_only()
def gemm_non_direct_acc(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    bias: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        bias_reg = tkw.read(bias, elements_per_thread=4)
        new_acc = tkw.exp2(bias_reg) + acc
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
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: %add_M:0_N:0_K:0
        # CHECK-SAME: [add](args = (%exp2_M:0_N:0_K:0, %acc_M:0_N:0_K:0), kwargs = {})
        # CHECK: %add_M:0_N:1_K:0
        # CHECK-SAME: [add](args = (%exp2_M:0_N:1_K:0, %acc_M:0_N:1_K:0), kwargs = {})
        # CHECK: %add_M:1_N:0_K:0
        # CHECK-SAME: [add](args = (%exp2_M:1_N:0_K:0, %acc_M:1_N:0_K:0), kwargs = {})
        # CHECK: %add_M:1_N:1_K:0
        # CHECK-SAME: [add](args = (%exp2_M:1_N:1_K:0, %acc_M:1_N:1_K:0), kwargs = {})
        # CHECK: %mma_M:0_N:0_K:0
        # CHECK-SAME: [mma](args = (%read_M:0_N:0_K:0, %read_1_M:0_N:0_K:0, %add_M:0_N:0_K:0, None), kwargs = {})
        # CHECK: %mma_M:0_N:0_K:1
        # CHECK-SAME: [mma](args = (%read_M:0_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:0_N:0_K:0, None), kwargs = {})
        # CHECK: %mma_M:0_N:1_K:0
        # CHECK-SAME: [mma](args = (%read_M:0_N:0_K:0, %read_1_M:0_N:1_K:0, %add_M:0_N:1_K:0, None), kwargs = {})
        # CHECK: %mma_M:0_N:1_K:1
        # CHECK-SAME: [mma](args = (%read_M:0_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:0_N:1_K:0, None), kwargs = {})
        # CHECK: %mma_M:1_N:0_K:0
        # CHECK-SAME: [mma](args = (%read_M:1_N:0_K:0, %read_1_M:0_N:0_K:0, %add_M:1_N:0_K:0, None), kwargs = {})
        # CHECK: %mma_M:1_N:0_K:1
        # CHECK-SAME: [mma](args = (%read_M:1_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:1_N:0_K:0, None), kwargs = {})
        # CHECK: %mma_M:1_N:1_K:0
        # CHECK-SAME: [mma](args = (%read_M:1_N:0_K:0, %read_1_M:0_N:1_K:0, %add_M:1_N:1_K:0, None), kwargs = {})
        # CHECK: %mma_M:1_N:1_K:1
        # CHECK-SAME: [mma](args = (%read_M:1_N:0_K:1, %read_1_M:0_N:1_K:1, %mma_M:1_N:1_K:0, None), kwargs = {})


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
            vector_shapes={M: 16, K: 64},
        )
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 64,
            BLOCK_K: 256,
        }
    ):
        graph = tiled_max()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: max(arg=[read_M:0_K:0, read_M:0_K:1, read_M:0_K:2, read_M:0_K:3], init=acc_M:0_K:0
        # CHECK: max(arg=[read_M:1_K:0, read_M:1_K:1, read_M:1_K:2, read_M:1_K:3], init=acc_M:1_K:0
        # CHECK: output(return_vals=([max_1_M:0_K:0, max_1_M:1_K:0],))
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
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: %a
        # CHECK-NEXT: %b
        # CHECK-NEXT: %c
        # CHECK-NEXT: %register_M:0_N:0_K:0
        # CHECK-NEXT: %reduction
        # CHECK-NEXT: %get_result_M:0_N:0_K:0
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register(shape=(M, N), dtype=f32, value=0.0, index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: reduction(axis=K, init_args=[register_M:0_N:0_K:0]
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: write(register_=get_result_M:0_N:0_K:0
        # CHECK-SAME: index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: output(return_vals=(None,))

        # Reduction subgraph:

        # CHECK: %acc_M:0_N:0_K:0

        # CHECK-NEXT: %a
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:0_N:0_K:1
        # CHECK-SAME: (%a, 4, None, (), None)

        # CHECK-NEXT: %b
        # CHECK-NEXT: %read_1_M:0_N:0_K:0
        # CHECK-SAME: (%b, 4, None, (), None)
        # CHECK-NEXT: %read_1_M:0_N:0_K:1
        # CHECK-SAME: (%b, 4, None, (), None)

        # CHECK-NEXT: %mma_M:0_N:0_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_1_M:0_N:0_K:0, %acc_M:0_N:0_K:0, None)
        # CHECK-NEXT: %mma_M:0_N:0_K:1
        # CHECK-SAME: (%read_M:0_N:0_K:1, %read_1_M:0_N:0_K:1, %mma_M:0_N:0_K:0, None)

        # CHECK-NEXT: return [mma_M:0_N:0_K:1]

        # CHECK: Custom format:

        # CHECK-NEXT: placeholder(_name=acc_M:0_N:0_K:0
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=4, mapping_dynamic_vals=(), index={N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:0 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-SAME: acc=acc_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: mma(lhs=read_M:0_N:0_K:1 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: rhs=read_1_M:0_N:0_K:1 (index = {N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K: ARGK*BLOCK_K + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-SAME: acc=mma_M:0_N:0_K:0 (index = {M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1}))
        # CHECK-NEXT: output(return_vals=([mma_M:0_N:0_K:1],))

        # CHECK-NEXT: -----


@tkw.wave_trace_only()
def attention(
    q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
    k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
    v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
    init_sum = tkl.Register[B, M, tkl.f32](0.0)
    init_max = tkl.Register[B, M, tkl.f32](-1e6)

    # This microkernel encodes the fact that if the reduction
    # dimension were tiled, then we would need to materialize a loop.
    @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
    def repeat(
        partial_max: tkl.Register[B, M, tkl.f32],
        partial_sum: tkl.Register[B, M, tkl.f32],
        acc: tkl.Register[B, N, M, tkl.f32],
    ) -> (
        tkl.Register[B, M, tkl.f32],
        tkl.Register[B, M, tkl.f32],
        tkl.Register[B, N, M, tkl.f32],
    ):
        imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=4)
        k_reg = tkw.read(k, elements_per_thread=4)
        inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
        x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
        m_j = tkw.max(x_j, partial_max, dim=K2)
        e_delta_max = tkw.exp2(partial_max - m_j)
        e_delta = tkw.exp2(x_j - m_j)
        e_init = partial_sum * e_delta_max
        d_j = tkw.sum(e_delta, e_init, dim=K2)
        imm_f16 = tkw.cast(e_delta, tkl.f16)
        v_reg = tkw.read(v, elements_per_thread=4)
        new_acc = acc * e_delta_max
        acc = tkw.mma(v_reg, imm_f16, new_acc)
        return m_j, d_j, acc

    # repeat represents the results of the loop
    res_max, res_sum, res_mm = repeat
    res = res_mm / res_sum
    tkw.write(res, c, elements_per_thread=4)


@run_test
def test_attention():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        )
    ]

    with tk.gen.TestLaunchContext(
        {
            K1: 64,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_B: 1,
            BLOCK_K2: 32,
        }
    ):
        graph = attention()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)

    # Root graph:
    # CHECK: write(register_=truediv_M:0_N:0_K2:0,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 4*floor((Mod($T0, 64))/16) : 4 : 16, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1})
    # CHECK: write(register_=truediv_M:0_N:1_K2:0,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1})
    # CHECK: write(register_=truediv_M:1_N:0_K2:0,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 4*floor((Mod($T0, 64))/16) : 4 : 16, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1})
    # CHECK: write(register_=truediv_M:1_N:1_K2:0,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1})

    # Reduction graph:
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
    # CHECK: read(memory=q,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, M: $T0*BLOCK_M/128 + $WG0*BLOCK_M + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})

    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
    # CHECK: read(memory=k,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, K2: ARGK*BLOCK_K2 + Mod($T0, 16) + 16 : 1 : 1, K1: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})

    # CHECK: read(memory=v,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K2: ARGK*BLOCK_K2 + 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=v,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) : 1 : 1, K2: ARGK*BLOCK_K2 + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
    # CHECK: read(memory=v,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K2: ARGK*BLOCK_K2 + 4*floor((Mod($T0, 64))/16) : 4 : 1})
    # CHECK: read(memory=v,
    # CHECK-SAME: index={B: $WG2*BLOCK_B : 1 : 1, N: $T1*BLOCK_N/2 + $WG1*BLOCK_N + Mod($T0, 16) + 16 : 1 : 1, K2: ARGK*BLOCK_K2 + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})


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
        infer_types(graph)
        add_get_results(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)
        # CHECK: %a
        # CHECK-NEXT: %c
        # CHECK-NEXT: %read_M:0_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %read_M:1_N:0_K:0
        # CHECK-SAME: (%a, 4, None, (), None)
        # CHECK-NEXT: %add_M:0_N:0_K:0
        # CHECK-SAME: (%read_M:0_N:0_K:0, %read_M:0_N:0_K:0)
        # CHECK-NEXT: %add_M:1_N:0_K:0
        # CHECK-SAME: (%read_M:1_N:0_K:0, %read_M:1_N:0_K:0)
        # CHECK-NEXT: %sub_M:0_N:0_K:0
        # CHECK-SAME: (%add_M:0_N:0_K:0, %read_M:0_N:0_K:0)
        # CHECK-NEXT: %sub_M:1_N:0_K:0
        # CHECK-SAME: (%add_M:1_N:0_K:0, %read_M:1_N:0_K:0)
        # CHECK-NEXT: %neg_M:0_N:0_K:0
        # CHECK-SAME: (%sub_M:0_N:0_K:0,)
        # CHECK-NEXT: %neg_M:1_N:0_K:0
        # CHECK-SAME: (%sub_M:1_N:0_K:0,)
        # CHECK-NEXT: %write_M:0_N:0_K:0
        # CHECK-SAME: (%neg_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:0_N:0_K:1
        # CHECK-SAME: (%neg_M:0_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:0
        # CHECK-SAME: (%neg_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: %write_M:1_N:0_K:1
        # CHECK-SAME: (%neg_M:1_N:0_K:0, %c, 4, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: read(memory=a, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: add(lhs=read_M:0_N:0_K:0, rhs=read_M:0_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: add(lhs=read_M:1_N:0_K:0, rhs=read_M:1_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: sub(lhs=add_M:0_N:0_K:0, rhs=read_M:0_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: sub(lhs=add_M:1_N:0_K:0, rhs=read_M:1_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: neg(arg=sub_M:0_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: neg(arg=sub_M:1_N:0_K:0, index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, N: $T1*BLOCK_N/4 + 4*$T1 + $WG1*BLOCK_N : 4 : 1}
        # CHECK-NEXT: write(register_=neg_M:0_N:0_K:0, memory=c, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, K: 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=neg_M:0_N:0_K:0, memory=c, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M : 1 : 16, K: 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=neg_M:1_N:0_K:0, memory=c, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, K: 4*$T2 + $WG2*BLOCK_K : 4 : 1}
        # CHECK-NEXT: write(register_=neg_M:1_N:0_K:0, memory=c, elements_per_thread=4, mapping_dynamic_vals=(), index={M: $T0*BLOCK_M/128 + $T0 + $WG0*BLOCK_M + 16 : 1 : 16, K: 4*$T2 + $WG2*BLOCK_K + 16 : 4 : 1}

        # CHECK: -----


@tkw.wave_trace_only()
def chained_gemm_32x32x8(
    q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
    k: tkl.Memory[B, K2, K1, SHARED_ADDRESS_SPACE, tkl.f16],
    v: tkl.Memory[B, N, K2, SHARED_ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

    @tkw.reduction(K2, init_args=[c_reg])
    def repeat(acc: tkl.Register[B, M, N, tkl.f32]) -> tkl.Register[B, M, N, tkl.f32]:
        inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=4)
        k_reg = tkw.read(k, elements_per_thread=4)
        kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
        qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
        qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
        v_reg = tkw.read(v, elements_per_thread=4)
        acc = tkw.mma(qk_cast_reg, v_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=16)


@run_test
def test_chained_gemm_32x32x8():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]

    mfma_variant = tkw.MMAType.F32_32x32x8_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    with tk.gen.TestLaunchContext(
        {
            BLOCK_B: 1,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            K1: 32,
        }
    ):
        graph = chained_gemm_32x32x8()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        print_trace(graph)

        # CHECK: %acc_M:0_N:0_K2:0
        # CHECK: %register
        # CHECK: %q
        # CHECK: %read_M:0_K2:0_K1:0
        # CHECK-SAME: (args = (%q, 4, None, (), None)
        # CHECK: %read_M:0_K2:0_K1:1
        # CHECK-SAME: (args = (%q, 4, None, (), None)
        # CHECK: %read_M:0_K2:0_K1:2
        # CHECK-SAME: (args = (%q, 4, None, (), None)
        # CHECK: %read_M:0_K2:0_K1:3
        # CHECK-SAME: (args = (%q, 4, None, (), None)
        # CHECK: %k
        # CHECK: %read_1_shared_M:0_K2:0_K1:0
        # CHECK-SAME: (args = (%k, 4, None, (), None)
        # CHECK: %read_1_shared_M:0_K2:0_K1:1
        # CHECK-SAME: (args = (%k, 4, None, (), None)
        # CHECK: %read_1_shared_M:0_K2:0_K1:2
        # CHECK-SAME: (args = (%k, 4, None, (), None)
        # CHECK: %read_1_shared_M:0_K2:0_K1:3
        # CHECK-SAME: (args = (%k, 4, None, (), None)
        # CHECK: %mma_M:0_K2:0_K1:0
        # CHECK-SAME: (args = (%read_1_shared_M:0_K2:0_K1:0, %read_M:0_K2:0_K1:0, %register_M:0_K2:0_K1:0, None)
        # CHECK: %mma_M:0_K2:0_K1:1
        # CHECK-SAME: (args = (%read_1_shared_M:0_K2:0_K1:1, %read_M:0_K2:0_K1:1, %mma_M:0_K2:0_K1:0, None)
        # CHECK: %mma_M:0_K2:0_K1:2
        # CHECK-SAME: (args = (%read_1_shared_M:0_K2:0_K1:2, %read_M:0_K2:0_K1:2, %mma_M:0_K2:0_K1:1, None)
        # CHECK: %mma_M:0_K2:0_K1:3
        # CHECK-SAME: (args = (%read_1_shared_M:0_K2:0_K1:3, %read_M:0_K2:0_K1:3, %mma_M:0_K2:0_K1:2, None)
        # CHECK: %permute_M:0_K2:0
        # CHECK-SAME: (args = (%mma_M:0_K2:0_K1:3, [B, M, K2])
        # CHECK: %cast_M:0_K2:0
        # CHECK-SAME: (args = (%permute_M:0_K2:0, f16)
        # CHECK: %v
        # CHECK: %read_2_shared_M:0_N:0_K2:0
        # CHECK-SAME: (args = (%v, 4, None, (), None)
        # CHECK: %read_2_shared_M:0_N:0_K2:1
        # CHECK-SAME: (args = (%v, 4, None, (), None)
        # CHECK: %read_2_shared_M:0_N:0_K2:2
        # CHECK-SAME: (args = (%v, 4, None, (), None)
        # CHECK: %read_2_shared_M:0_N:0_K2:3
        # CHECK-SAME: (args = (%v, 4, None, (), None)
        # CHECK: %reshape_M:0_N:0_K2:0
        # CHECK-SAME: (args = ([%cast_M:0_K2:0], {K2: 32, M: 32, K1: 8, B: 0})
        # CHECK: %reshape_M:0_N:0_K2:1
        # CHECK-SAME: (args = ([%cast_M:0_K2:0], {K2: 32, M: 32, K1: 8, B: 0})
        # CHECK: %reshape_M:0_N:0_K2:2
        # CHECK-SAME: (args = ([%cast_M:0_K2:0], {K2: 32, M: 32, K1: 8, B: 0})
        # CHECK: %reshape_M:0_N:0_K2:3
        # CHECK-SAME: (args = ([%cast_M:0_K2:0], {K2: 32, M: 32, K1: 8, B: 0})
        # CHECK: %mma_1_M:0_N:0_K2:0
        # CHECK-SAME: (args = (%reshape_M:0_N:0_K2:0, %read_2_shared_M:0_N:0_K2:0, %acc_M:0_N:0_K2:0, None)
        # CHECK: %mma_1_M:0_N:0_K2:1
        # CHECK-SAME: (args = (%reshape_M:0_N:0_K2:1, %read_2_shared_M:0_N:0_K2:1, %mma_1_M:0_N:0_K2:0, None)
        # CHECK: %mma_1_M:0_N:0_K2:2
        # CHECK-SAME: (args = (%reshape_M:0_N:0_K2:2, %read_2_shared_M:0_N:0_K2:2, %mma_1_M:0_N:0_K2:1, None)
        # CHECK: %mma_1_M:0_N:0_K2:3
        # CHECK-SAME: (args = (%reshape_M:0_N:0_K2:3, %read_2_shared_M:0_N:0_K2:3, %mma_1_M:0_N:0_K2:2, None)
        # CHECK: return [mma_1_M:0_N:0_K2:3]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
