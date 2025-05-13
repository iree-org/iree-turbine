# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.promotion import promote_placeholders
from iree.turbine.kernel.wave.hoisting import hoist_loop_invariant_ops
from iree.turbine.kernel.wave.expansion.expansion import expand_graph
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.unrolling import unroll
from iree.turbine.kernel.wave.utils.print_utils import print_graph
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
from iree.turbine.kernel.ops.wave_ops import Iterate, get_custom
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.expansion.expansion import add_get_results
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.utils.graph_utils import initialize_iter_args
from iree.turbine.kernel.wave.minimize_global_loads import minimize_global_loads
from iree.turbine.kernel.wave.shared_memory_indexing import (
    apply_shared_memory_indexing_corrections,
)
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)


M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0
ARGK = tkl.sym.ARGK


@run_test
def test_iteration():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0, M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.TilingConstraint(B, BLOCK_B)]

    # This test demonstrates a batched GEMM operation using Wave's iteration feature.
    # The kernel performs matrix multiplication C[b,m,n] = A[b,m,k] * B[b,n,k] for each batch b.
    #
    # The iteration is done over the batch dimension B. The number of iterations
    # is determined by the tiling constraint on B.
    @tkw.wave(constraints)
    def iterated_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        @tkw.iterate(B, init_args=[])
        def body():

            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            B: 10,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    iterated_gemm_kernel = wave_compile(options, iterated_gemm)
    print(iterated_gemm_kernel.asm)

    # CHECK-DAG:            %[[C10:.*]] = arith.constant 10 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG:            %[[C1:.*]] = arith.constant 1 : index
    # CHECK-COUNT-1:        memref.alloc
    # CHECK:                scf.for %[[ARG3:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
    # CHECK:                    amdgpu.lds_barrier
    # CHECK:                    scf.for %arg4 = %[[C0]] to %[[C4]] step %[[C1]]
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                vector.load
    # CHECK:                        amdgpu.mfma
    # CHECK:                        scf.yield
    # CHECK-COUNT-4:            vector.store


@run_test
def test_iteration_with_condition():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0, M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    # B is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(B)]

    @tkw.wave(constraints)
    def iterated_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
        init_value: tkl.i32,  # type: ignore
    ):
        @tkw.iterate(B, start=init_value, condition=B < 10, init_args=[])
        def body():
            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(B, next_value)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            B: 10,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    iterated_gemm = wave_compile(options, iterated_gemm)
    print(iterated_gemm.asm)

    # CHECK-DAG:            %[[C10:.*]] = arith.constant 10 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG:            %[[C1:.*]] = arith.constant 1 : index
    # CHECK-COUNT-1:        memref.alloc
    # CHECK-DAG:            %[[INIT_B:.*]] = arith.index_cast
    # CHECK:                scf.while (%[[ARG:.*]] = %[[INIT_B]]) : (index) -> index {
    # CHECK:                   %[[COND:.*]] = arith.cmpi slt, %[[ARG]], %[[C10]] : index
    # CHECK:                   scf.condition(%[[COND]]) %[[ARG]] : index
    # CHECK:                } do {
    # CHECK:                 ^bb0(%[[ARG:.*]]: index):
    # CHECK:                    amdgpu.lds_barrier
    # CHECK:                    %[[ARG4:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[CST:.*]]) -> (vector<4xf32>) {
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                vector.load
    # CHECK:                        amdgpu.mfma
    # CHECK:                        scf.yield
    # CHECK-COUNT-4:            vector.store


@run_test
def test_iteration_with_condition_and_init_value():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0, M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    # B is iterated over and so we define a tiling constraint on it.
    # However, there is no notion of tile size for the iteration as
    # it is an unstructured loop.
    constraints += [tkw.TilingConstraint(B)]

    @tkw.wave(constraints)
    def iterated_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
        output_sum: tkl.Memory[N, M, ADDRESS_SPACE_0, tkl.f32],
        init_value: tkl.i32,  # type: ignore
    ):

        o_reg = tkl.Register[N, M, tkl.f32](0.0)

        @tkw.iterate(B, start=init_value, condition=B < 10, init_args=[o_reg])
        def body(outer_acc: tkl.Register[N, M, tkl.f32]):
            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(K, init_args=[c_reg])
            def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)
            permuted = tkw.permute(repeat, target_shape=[N, M])
            outer_acc += permuted

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(B, next_value)

            return outer_acc

        tkw.write(body, output_sum, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            B: 10,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    iterated_gemm = wave_compile(options, iterated_gemm)
    print(iterated_gemm.asm)

    # CHECK-DAG:            %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:            %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[C10:.*]] = arith.constant 10 : index
    # CHECK-DAG:            %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-COUNT-1:        memref.alloc
    # CHECK:                %[[CAST_INIT_B:.*]] = arith.index_cast
    # CHECK:                %[[WHILE:.*]] = scf.while (%[[ACC:.*]] = %[[CST_0]], %[[B:.*]] = %[[CAST_INIT_B]]) : (vector<4xf32>, index) -> (vector<4xf32>, index) {
    # CHECK:                    %[[COND:.*]] = arith.cmpi slt, %[[B]], %[[C10]] : index
    # CHECK:                    scf.condition(%[[COND]]) %[[ACC]], %[[B]] : vector<4xf32>, index
    # CHECK:                } do {
    # CHECK:                ^bb0(%[[ACC:.*]]: vector<4xf32>, %[[B:.*]]: index):
    # CHECK:                    amdgpu.lds_barrier
    # CHECK:                    %[[FOR:.*]] = scf.for %[[ARG7:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG8:.*]] = %[[CST_0]]) -> (vector<4xf32>) {
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                vector.load
    # CHECK:                        amdgpu.mfma
    # CHECK:                        scf.yield
    # CHECK-COUNT-4:                vector.store


@run_test
def test_partial_unroll_iteration():
    # This tests partial unrolling of an iteration.
    @tkw.wave_trace_only()
    def iterated_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]

    with tk.gen.TestLaunchContext(
        {
            M: 64,
            N: 128,
            K: 128,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
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
        },
    ):
        trace: CapturedTrace = iterated_gemm()
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

        # Check the graph before unrolling
        # Find iterate and unroll
        iterate = get_custom(
            trace.walk(lambda node: isinstance(get_custom(node), Iterate))[0]
        )
        assert isinstance(iterate, Iterate)
        print_graph(trace.get_subgraph(iterate.subgraph_name))

        # CHECK: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: return

        # Partially Unroll the iterate
        unroll(iterate, 2, trace, constraints)
        print_graph(trace.get_subgraph(iterate.subgraph_name))
        assert iterate.count == 4
        assert iterate.step == 2

        # TODO: Check that the bounds are correct, and steps

        # CHECK: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: return

        # Unroll the unrolled iterate again
        unroll(iterate, 2, trace, constraints)
        print_graph(trace.get_subgraph(iterate.subgraph_name))
        assert iterate.count == 2
        assert iterate.step == 4

        # CHECK: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: placeholder
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [write]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [read]
        # CHECK-NEXT: [mma]
        # CHECK-NEXT: return
