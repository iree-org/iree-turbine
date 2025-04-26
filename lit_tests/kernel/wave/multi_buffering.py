# RUN: python %s | FileCheck %s

import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.promotion import promote_placeholders
from iree.turbine.kernel.wave.hoisting import hoist_loop_invariant_ops
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.lang.global_symbols import (
    GLOBAL_ADDRESS_SPACE,
    SHARED_ADDRESS_SPACE,
    READ_SHARED_DELAY,
    WRITE_SHARED_DELAY,
    READ_GLOBAL_DELAY,
    WRITE_GLOBAL_DELAY,
    MMA_DELAY,
    SHARED_MEMORY_UNITS,
    GLOBAL_MEMORY_UNITS,
    MMA_UNITS,
    VALU_DELAY,
    VALU_UNITS,
    SHUFFLE_DELAY,
    SHUFFLE_UNITS,
)
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.ops.wave_ops import (
    Allocate,
    Read,
    Iterate,
    Write,
    get_custom,
    GetResult,
)
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.utils.graph_utils import initialize_iter_args
from iree.turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from iree.turbine.kernel.wave.shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
)
from iree.turbine.kernel.wave.scheduling.schedule import schedule_graph, SchedulingType
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
import torch.fx as fx

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

    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run_test
def test_gemm_multibuffering():
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
            VALU_DELAY: 1,
            VALU_UNITS: 2,
            SHUFFLE_DELAY: 1,
            SHUFFLE_UNITS: 2,
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
        schedule_graph(
            trace,
            constraints,
            True,
            scheduling_type=SchedulingType.MODULO_MULTI_BUFFERED,
        )

        def print_affected_node(node: fx.Node):
            match custom := get_custom(node):
                case Allocate():
                    print(custom)
                case Read() | Write():
                    if custom.memory_type.address_space == SHARED_ADDRESS_SPACE:
                        print(custom)
                case Iterate():
                    print("reduction begin")
                    for node in trace.get_subgraph(custom.subgraph_name).nodes:
                        print_affected_node(node)
                    print("reduction end")
                case _:
                    pass

        for node in trace.get_root_graph().nodes:
            print_affected_node(node)

        # CHECK: allocate(shape=(2*N, K), distributed_shape=(2*BLOCK_N, BLOCK_K + 4)
        # CHECK-NEXT: allocate(shape=(2*M, K), distributed_shape=(2*BLOCK_M, BLOCK_K + 4)
        # CHECK-NEXT: write(register_=read_21,
        # CHECK-NEXT: write(register_=read_22,
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-NEXT: read(memory=allocate,
        # CHECK: reduction begin
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-SAME: {N: BLOCK_N*(Mod($ARGK, 2)) + BLOCK_N/2 + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1}
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-SAME: index={M: BLOCK_M*(Mod($ARGK, 2)) + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1}
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-SAME: {N: BLOCK_N*(Mod($ARGK, 2)) + BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1}
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-SAME: {N: BLOCK_N*(Mod($ARGK, 2)) + BLOCK_N/2 + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1}
        # CHECK-NEXT: write(register_=read_21,
        # CHECK-SAME: index={M: xor(BLOCK_M*(Mod($ARGK, 2)), BLOCK_M) + Mod(32*$T1 + floor($T0/4), 64) : 1 : 1, K: 8*(Mod($T0, 4)) : 8 : 1}
        # CHECK-NEXT: write(register_=read_22,
        # CHECK-SAME: index={N: BLOCK_N/2 + xor(BLOCK_N*(Mod($ARGK, 2)), BLOCK_N) + Mod(32*$T1 + floor($T0/4), 64) : 1 : 1, K: 8*(Mod($T0, 4)) : 8 : 1}
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-SAME: index={M: xor(BLOCK_M*(Mod($ARGK, 2)), BLOCK_M) + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) : 4 : 1}
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-SAME: index={M: xor(BLOCK_M*(Mod($ARGK, 2)), BLOCK_M) + Mod($T0, 16) + 16 : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1}
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-SAME: index={N: BLOCK_N/2 + xor(BLOCK_N*(Mod($ARGK, 2)), BLOCK_N) + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1}
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-SAME: index={M: xor(BLOCK_M*(Mod($ARGK, 2)), BLOCK_M) + Mod($T0, 16) : 1 : 1, K: 4*floor((Mod($T0, 64))/16) + 16 : 4 : 1}
        # CHECK-NEXT: reduction end
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-NEXT: read(memory=allocate,
        # CHECK-NEXT: read(memory=allocate_1,
        # CHECK-NEXT: read(memory=allocate_1,


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
