# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.promotion import promote_placeholders
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
from iree.turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from iree.turbine.kernel.wave.shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
)
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.wave.analysis.partition_strided_operators import (
    partition_strided_operators,
)


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
        IndexingContext.current().finalize()
        initialize_iter_args(trace)
        add_get_results(trace)
        infer_types(trace)
        promote_placeholders(trace, constraints)
        set_node_indices(trace, constraints)
        expand_graph(trace, constraints)
        set_post_expansion_indices(trace, constraints)
        hoist_loop_invariant_ops(trace, constraints)
        minimize_global_loads(trace, constraints)
        apply_shared_memory_indexing_corrections(trace, constraints)
        partition_strided_operators(trace, constraints)
        print_trace(trace)
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
        # CHECK-SAME (K, [%register_M:0_N:0_K:0, %register_M:0_N:1_K:0, %register_M:1_N:0_K:0, %register_M:1_N:1_K:0]
        # CHECK-NEXT: %get_result_M:0_N:0_K:0
        # CHECK-SAME: (%reduction, 0)
        # CHECK-NEXT: %get_result_M:0_N:1_K:0
        # CHECK-SAME: (%reduction, 1)
        # CHECK-NEXT: %get_result_M:1_N:0_K:0
        # CHECK-SAME: (%reduction, 2)
        # CHECK-NEXT: %get_result_M:1_N:1_K:0
        # CHECK-SAME: (%reduction, 3)
        # CHECK-NEXT: extract_slice
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, [0], [1], [1])
        # CHECK-NEXT: %write_5
        # CHECK-SAME: (%extract_slice, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_1
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, [1], [1], [1])
        # CHECK-NEXT: %write_6
        # CHECK-SAME: (%extract_slice_1, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_2
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, [2], [1], [1])
        # CHECK-NEXT: %write_7
        # CHECK-SAME: (%extract_slice_2, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_3
        # CHECK-SAME: (%get_result_M:0_N:0_K:0, [3], [1], [1])
        # CHECK-NEXT: %write_8
        # CHECK-SAME: (%extract_slice_3, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_4
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, [0], [1], [1])
        # CHECK-NEXT: %write_9
        # CHECK-SAME: (%extract_slice_4, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_5
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, [1], [1], [1])
        # CHECK-NEXT: %write_10
        # CHECK-SAME: (%extract_slice_5, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_6
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, [2], [1], [1])
        # CHECK-NEXT: %write_11
        # CHECK-SAME: (%extract_slice_6, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_7
        # CHECK-SAME: (%get_result_M:0_N:1_K:0, [3], [1], [1])
        # CHECK-NEXT: %write_12
        # CHECK-SAME: (%extract_slice_7, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_8
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, [0], [1], [1])
        # CHECK-NEXT: %write_13
        # CHECK-SAME: (%extract_slice_8, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_9
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, [1], [1], [1])
        # CHECK-NEXT: %write_14
        # CHECK-SAME: (%extract_slice_9, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_10
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, [2], [1], [1])
        # CHECK-NEXT: %write_15
        # CHECK-SAME: (%extract_slice_10, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_11
        # CHECK-SAME: (%get_result_M:1_N:0_K:0, [3], [1], [1])
        # CHECK-NEXT: %write_16
        # CHECK-SAME: (%extract_slice_11, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_12
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, [0], [1], [1])
        # CHECK-NEXT: %write_17
        # CHECK-SAME: (%extract_slice_12, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_13
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, [1], [1], [1])
        # CHECK-NEXT: %write_18
        # CHECK-SAME: (%extract_slice_13, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_14
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, [2], [1], [1])
        # CHECK-NEXT: %write_19
        # CHECK-SAME: (%extract_slice_14, %c, 1, None, ())
        # CHECK-NEXT: extract_slice_15
        # CHECK-SAME: (%get_result_M:1_N:1_K:0, [3], [1], [1])
        # CHECK-NEXT: %write_20
        # CHECK-SAME: (%extract_slice_15, %c, 1, None, ())
        # CHECK-NEXT: return None

        # CHECK: Custom format:
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=b
        # CHECK-NEXT: placeholder(_name=c
        # CHECK-NEXT: register
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $WG1*BLOCK_N + BLOCK_N/2 + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) : 4 : 16, N: $WG1*BLOCK_N + BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $WG1*BLOCK_N + BLOCK_N/2 + Mod($T0, 16) : 1 : 1})
        # CHECK-NEXT: register(
        # CHECK-SAME: index={M: $WG0*BLOCK_M + 4*floor((Mod($T0, 64))/16) + 16 : 4 : 16, N: $WG1*BLOCK_N + BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1})
        # CHECK-NEXT: allocate(
        # CHECK-NEXT: allocate(
        # CHECK-NEXT: reduction(
        # CHECK-NEXT: get_result(value=reduction, res_idx=0)
        # CHECK-NEXT: get_result(value=reduction, res_idx=1)
        # CHECK-NEXT: get_result(value=reduction, res_idx=2)
        # CHECK-NEXT: get_result(value=reduction, res_idx=3)
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:0_K:0, offset=[0], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:0_K:0, offset=[1], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_1, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 1 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:0_K:0, offset=[2], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_2, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 2 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:0_K:0, offset=[3], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_3, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 3 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:1_K:0, offset=[0], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_4, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:1_K:0, offset=[1], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_5, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 1 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:1_K:0, offset=[2], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_6, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 2 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:0_N:1_K:0, offset=[3], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_7, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 3 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:0_K:0, offset=[0], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_8, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 16 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:0_K:0, offset=[1], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_9, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 17 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:0_K:0, offset=[2], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_10, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 18 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:0_K:0, offset=[3], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_11, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 19 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 32 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:1_K:0, offset=[0], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_12, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 16 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:1_K:0, offset=[1], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_13, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 17 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:1_K:0, offset=[2], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_14, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 18 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})
        # CHECK-NEXT: extract_slice(register_=get_result_M:1_N:1_K:0, offset=[3], size=[1], stride=[1])
        # CHECK-NEXT: write(register_=extract_slice_15, memory=c, elements_per_thread=1,
        # CHECK-SAME: index={M: 64*$WG0 + 4*floor((Mod($T0, 64))/16) + 19 : 1 : 1, N: 64*$WG1 + Mod($T0, 16) + 48 : 1 : 1})

        # Reduction subgraph:
        # CHECK: %b
        # CHECK-NEXT: %a
        # CHECK-NEXT: %acc_M:0_N:0_K:0
        # CHECK-NEXT: %acc_M:0_N:1_K:0
        # CHECK-NEXT: %acc_M:1_N:0_K:0
        # CHECK-NEXT: %acc_M:1_N:1_K:0
        # CHECK-NEXT: %read_37
        # CHECK-SAME: (%a, 8, None, (), None)
        # CHECK-NEXT: %write_18
        # CHECK-SAME: (%read_37, %allocate, 8, None, ())
        # CHECK-NEXT: %read_38
        # CHECK-SAME: (%a, 8, None, (), None)
        # CHECK-NEXT: %write_19
        # CHECK-SAME: (%read_38, %allocate, 8, None, ())
        # CHECK-NEXT: %read_39
        # CHECK-SAME: (%b, 8, None, (), None)
        # CHECK-NEXT: %write_20
        # CHECK-SAME: (%read_39, %allocate_1, 8, None, ())
        # CHECK-NEXT: %read_40
        # CHECK-SAME: (%b, 8, None, (), None)
        # CHECK-NEXT: %write_21
        # CHECK-SAME: (%read_40, %allocate_1, 8, None, ())
        # CHECK-NEXT: %read_4_shared_M:0_N:0_K:0
        # CHECK-NEXT: %read_4_shared_M:0_N:0_K:1
        # CHECK-NEXT: %read_4_shared_M:0_N:0_K:2
        # CHECK-NEXT: %read_4_shared_M:0_N:0_K:3
        # CHECK-NEXT: %read_4_shared_M:0_N:1_K:0
        # CHECK-NEXT: %read_4_shared_M:0_N:1_K:1
        # CHECK-NEXT: %read_4_shared_M:0_N:1_K:2
        # CHECK-NEXT: %read_4_shared_M:0_N:1_K:3
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:0
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:1
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:2
        # CHECK-NEXT: %read_2_shared_M:0_N:0_K:3
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:0
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:1
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:2
        # CHECK-NEXT: %read_2_shared_M:1_N:0_K:3
        # CHECK-NEXT: %mma_M:0_N:0_K:0
        # CHECK-NEXT: %mma_M:0_N:0_K:1
        # CHECK-NEXT: %mma_M:0_N:0_K:2
        # CHECK-NEXT: %mma_M:0_N:0_K:3
        # CHECK-NEXT: %mma_M:0_N:1_K:0
        # CHECK-NEXT: %mma_M:0_N:1_K:1
        # CHECK-NEXT: %mma_M:0_N:1_K:2
        # CHECK-NEXT: %mma_M:0_N:1_K:3
        # CHECK-NEXT: %mma_M:1_N:0_K:0
        # CHECK-NEXT: %mma_M:1_N:0_K:1
        # CHECK-NEXT: %mma_M:1_N:0_K:2
        # CHECK-NEXT: %mma_M:1_N:0_K:3
        # CHECK-NEXT: %mma_M:1_N:1_K:0
        # CHECK-NEXT: %mma_M:1_N:1_K:1
        # CHECK-NEXT: %mma_M:1_N:1_K:2
        # CHECK-NEXT: %mma_M:1_N:1_K:3

        # Reduction subgraph (custom format):
        # CHECK: Custom format:
        # CHECK: placeholder(_name=b, _type=Memory[N, K].of(f16))
        # CHECK-NEXT: placeholder(_name=a
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:0_N:1_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:0_K:0
        # CHECK-NEXT: placeholder(_name=acc_M:1_N:1_K:0
        # CHECK-NEXT: read(memory=a, elements_per_thread=8,
        # CHECK-SAME: index={M: $WG0*BLOCK_M + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 1 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_37, memory=allocate, elements_per_thread=8,
        # CHECK-SAME: index={M: Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 1 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=a, elements_per_thread=8,
        # CHECK-SAME: index={M: $WG0*BLOCK_M + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 1 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_38, memory=allocate, elements_per_thread=8,
        # CHECK-SAME: index={M: Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 1 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=8,
        # CHECK-SAME: index={N: $WG1*BLOCK_N + BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 1 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_39, memory=allocate_1, elements_per_thread=8,
        # CHECK-SAME: index={N: BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8), 64) : 1 : 1, K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=b, elements_per_thread=8,
        # CHECK-SAME: index={N: $WG1*BLOCK_N + BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64) : 1 : 1, K: ARGK*BLOCK_K + 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: write(register_=read_40, memory=allocate_1, elements_per_thread=8,
        # CHECK-SMAE: index={N: BLOCK_N/2 + Mod(16*$T1 + 32*$T2 + floor($T0/8) + 32, 64), K: 8*(Mod($T0, 8)) : 8 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate_1, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_20, write_21], index={N: BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 32 : 4 : 1})
        # CHECK-NEXT: read(memory=allocate, elements_per_thread=4, mapping_dynamic_vals=(), _write_dependency=[write_18, write_19], index={M: Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 48 : 4 : 1})


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
