# RUN: python %s | FileCheck %s

from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
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


@run_test
def test_iteration():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 1, M: 16, N: 16, K: 16},
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
    iterated_gemm = wave_compile(options, iterated_gemm)
    print(iterated_gemm.asm)

    # CHECK-DAG:            %[[C10:.*]] = arith.constant 10 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[C4:.*]] = arith.constant 4 : index
    # CHECK-DAG:            %[[C1:.*]] = arith.constant 1 : index
    # CHECK:                scf.for %[[ARG3:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
    # CHECK:                    amdgpu.lds_barrier
    # CHECK:                    scf.for %arg4 = %[[C0]] to %[[C4]] step %[[C1]]
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                memref.alloc
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
            vector_shapes={B: 1, M: 16, N: 16, K: 16},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    # B is tiled with user-defined init and next symbols and terminates
    # when the user defined condition is met.
    INIT_B = tkl.sym.INIT_B
    NEXT_B = tkl.sym.NEXT_B
    constraints += [
        tkw.TilingConstraint(
            B, init_symbol=INIT_B, next_symbol=NEXT_B, condition=lambda x: x < 10
        )
    ]

    @tkw.wave(constraints)
    def iterated_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
        init_value: tkl.i32,
    ):
        # Set the initial value for the iteration.
        tkw.set_symbol(INIT_B, init_value)

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

            # Set the next value for the iteration.
            # In this case, we are using a simple increment operation,
            # but this can be replaced with any other operation.
            index_b = tkw.self_index(B, tkl.i32)
            next_value = tkw.apply_expr(index_b, lambda x: x + 1)
            tkw.set_symbol(NEXT_B, next_value)

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
    # CHECK-DAG:            %[[INIT_B:.*]] = vector.extractelement %1[] : vector<index>
    # CHECK:                scf.while (%[[ARG:.*]] = %[[INIT_B]]) : (index) -> index {
    # CHECK:                   %[[COND:.*]] = arith.cmpi slt, %[[ARG]], %[[C10]] : index
    # CHECK:                   scf.condition(%[[COND]]) %[[ARG]] : index
    # CHECK:                } do {
    # CHECK:                 ^bb0(%[[ARG:.*]]: index):
    # CHECK:                    amdgpu.lds_barrier
    # CHECK:                    %[[ARG4:.*]] = scf.for %[[ARG5:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ARG6:.*]] = %[[CST:.*]]) -> (vector<4xf32>) {
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                memref.alloc
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK-COUNT-1:                vector.load
    # CHECK-COUNT-1:                vector.store
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-2:                vector.load
    # CHECK:                        amdgpu.mfma
    # CHECK:                        scf.yield
    # CHECK-COUNT-4:            vector.store
