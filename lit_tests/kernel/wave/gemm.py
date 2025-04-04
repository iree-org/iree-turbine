# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
B_KV = tkl.sym.B_KV
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def get_wave_compile_options(canonicalize: bool = False, dynamic_symbols=[]):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return WaveCompileOptions(
        subs=bindings,
        canonicalize=canonicalize,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
    )


@run_test
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
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
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL:    test_gemm
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @gemm
    # CHECK-COUNT-2     memref.alloc()
    # CHECK:            scf.for
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-2:      vector.load
    # CHECK:              amdgpu.mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def test_gemm_bias():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_bias(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        bias_reg = tkw.read(bias)
        result = repeat + bias_reg
        tkw.write(result, c)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    gemm_bias = wave_compile(options, gemm_bias)
    print(gemm_bias.asm)

    # CHECK-LABEL:     func.func @gemm_bias
    # CHECK:             scf.for
    # CHECK-COUNT-1:       vector.load
    # CHECK-COUNT-1:       vector.store
    # CHECK-COUNT-1:       vector.load
    # CHECK-COUNT-1:       vector.store
    # CHECK-COUNT-2:       vector.load
    # CHECK-COUNT-1:       amdgpu.mfma
    # CHECK:             scf.yield
    # Load transposed bias from global
    # CHECK-COUNT-4:       vector.load
    # Add result to bias
    # CHECK-COUNT-1:       arith.addf
    # Store result
    # CHECK-COUNT-4:       vector.store


@run_test
def test_cdna2_int_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.I32_16x16x16_I8,
        )
    ]

    @tkw.wave(constraints)
    def cdna2_int_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.f32]:
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
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    cdna2_int_gemm = wave_compile(options, cdna2_int_gemm)
    print(cdna2_int_gemm.asm)

    # CHECK-LABEL:    func.func @cdna2_int_gemm
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
    # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0> : vector<4xi32>
    # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x24xi8, #gpu.address_space<workgroup>>
    # CHECK:            %[[ALLOC_1:.+]] = memref.alloc() : memref<32x24xi8, #gpu.address_space<workgroup>>
    # CHECK:            %[[GLOBAL_1:.+]] = stream.binding.subspan %[[ARG1]]
    # CHECK:            %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
    # CHECK:            scf.for %[[IVAR:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[CST]]) -> (vector<4xi32>) {
    # CHECK:                %[[REG_0:.+]] = vector.load %[[GLOBAL_0]]
    # CHECK:                vector.store %[[REG_0]], %[[ALLOC_1]]
    # CHECK:                %[[REG_1:.+]] = vector.load %[[GLOBAL_1]]
    # CHECK:                vector.store %[[REG_1]], %[[ALLOC_0]]
    # CHECK:                %[[RHS:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<32x24xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK:                %[[LHS:.+]] = vector.load %[[ALLOC_1]]{{.*}} : memref<32x24xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK:                %[[MMA:.+]] = amdgpu.mfma %[[LHS]] * %[[RHS]] + %[[ACC]]  {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xi8>, vector<4xi8>, vector<4xi32>
    # CHECK:                scf.yield %[[MMA]] : vector<4xi32>


@run_test
def test_cdna3_int_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mma_variant = tkw.MMAType.I32_16x16x32_I8
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mma_variant,
        )
    ]

    @tkw.wave(constraints)
    def cdna3_int_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.i32],
    ):
        c_reg = tkl.Register[M, N, tkl.i32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.i32]) -> tkl.Register[M, N, tkl.f32]:
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
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mma_variant),
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    cdna3_int_gemm = wave_compile(options, cdna3_int_gemm)
    print(cdna3_int_gemm.asm)

    # CHECK-LABEL:    func.func @cdna3_int_gemm
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
    # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
    # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0> : vector<4xi32>
    # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x40xi8, #gpu.address_space<workgroup>>
    # CHECK:            %[[ALLOC_1:.+]] = memref.alloc() : memref<32x40xi8, #gpu.address_space<workgroup>>
    # CHECK:            %[[GLOBAL_1:.+]] = stream.binding.subspan %[[ARG1]]
    # CHECK:            %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
    # CHECK:            scf.for %[[IVAR:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[CST]]) -> (vector<4xi32>) {
    # CHECK:                %[[REG_0:.+]] = vector.load %[[GLOBAL_0]]
    # CHECK:                vector.store %[[REG_0]], %[[ALLOC_1]]
    # CHECK:                %[[REG_1:.+]] = vector.load %[[GLOBAL_1]]
    # CHECK:                vector.store %[[REG_1]], %[[ALLOC_0]]
    # CHECK:                %[[RHS:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<32x40xi8, #gpu.address_space<workgroup>>, vector<8xi8>
    # CHECK:                %[[LHS:.+]] = vector.load %[[ALLOC_1]]{{.*}} : memref<32x40xi8, #gpu.address_space<workgroup>>, vector<8xi8>
    # CHECK:                %[[MMA:.+]] = amdgpu.mfma %[[LHS]] * %[[RHS]] + %[[ACC]]  {blocks = 1 : i32, k = 32 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<8xi8>, vector<8xi8>, vector<4xi32>
    # CHECK:                scf.yield %[[MMA]] : vector<4xi32>


@run_test
def test_batched_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            B: 12,
            M: 64,
            N: 128,
            K: 64,
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
    batched_gemm = wave_compile(options, batched_gemm)
    print(batched_gemm.asm)

    # CHECK-LABEL:    test_batched_gemm
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @batched_gemm
    # CHECK-COUNT-2     memref.alloc()
    # CHECK:            scf.for
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-2:      vector.load
    # CHECK:              amdgpu.mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def test_chained_gemm():
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def chained_gemm(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE_0, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K1: 32,
            K2: 256,
            B: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    chained_gemm = wave_compile(options, chained_gemm)
    print(chained_gemm.asm)

    # CHECK-LABEL:     func.func @chained_gemm
    # CHECK-SAME:        (%[[ARG0:.*]]: !stream.binding, %{{.+}}: !stream.binding, %{{.+}}: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK:             %[[ALLOC:.+]] = memref.alloc() : memref<1x32x36xf16, #gpu.address_space<workgroup>>
    # CHECK:             %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
    # CHECK-COUNT-4:     vector.load %[[GLOBAL_0]]
    # CHECK:             {{.*}} = scf.for
    # CHECK-COUNT-4:       {{.*}} = vector.load %[[ALLOC]]
    # CHECK-COUNT-8:       {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:       {{.*}} = arith.truncf
    # CHECK-COUNT-8:       {{.*}} = amdgpu.mfma
    # CHECK:             scf.yield


@run_test
def test_chained_gemm_32x32x8():
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_32x32x8_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def chained_gemm_32x32x8(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(qk_cast_reg, v_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K1: 32,
            K2: 256,
            B: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )

    chained_gemm_32x32x8 = wave_compile(options, chained_gemm_32x32x8)
    print(chained_gemm_32x32x8.asm)

    # CHECK-LABEL:     func.func @chained_gemm_32x32x8
    # CHECK-SAME:        (%[[ARG0:.*]]: !stream.binding, %{{.+}}: !stream.binding, %{{.+}}: !stream.binding, %{{.+}}: !stream.binding)
    # CHECK:             %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
    # CHECK:             %[[GLOBAL_READ_0:.+]] = vector.load %[[GLOBAL_0]]
    # CHECK:             {{.*}} = scf.for
    # CHECK-COUNT-4:       {{.*}} = amdgpu.mfma
    # CHECK-COUNT-1:       {{.*}} = arith.truncf
    # CHECK:               {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [4], strides = [1]}
    # CHECK:               {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [4], sizes = [4], strides = [1]}
    # CHECK:               {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [8], sizes = [4], strides = [1]}
    # CHECK:               {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [12], sizes = [4], strides = [1]}
    # CHECK-COUNT-4:       {{.*}} = amdgpu.mfma
    # CHECK:             scf.yield


@run_test
def test_chained_gemm_32x32x16():
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = [tkw.MMAType.F32_32x32x16_F8, tkw.MMAType.F32_32x32x16_K4_F8]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def chained_gemm_32x32x16(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(qk_cast_reg, v_reg, acc, mfma_variant[1])
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K1: 32,
            K2: 256,
            B: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )

    chained_gemm_32x32x16 = wave_compile(options, chained_gemm_32x32x16)
    print(chained_gemm_32x32x16.asm)

    # CHECK-LABEL:     func.func @chained_gemm_32x32x16(
    # CHECK:             %[[V_SHARED:.+]] = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
    # CHECK:             {{.*}} = scf.for

    # Loading V from shared memory with interleaved/k-width=4, then using insert slice to combine them together.
    # This is to align V's layout with the layout of 1st MMA output.
    # CHECK-COUNT-2:       %[[V_REG_0:.+]] = vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
    # CHECK-COUNT-2:       %[[V_REG_1:.+]] = vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
    # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
    # CHECK:               %[[V_REG_F8_0:.+]] = arith.truncf %[[V_REG_0]] : vector<8xf16> to vector<8xf8E4M3FNUZ>
    # CHECK:               %[[V_REG_F8_1:.+]] = arith.truncf %[[V_REG_1]] : vector<8xf16> to vector<8xf8E4M3FNUZ>

    # 2nd MMA
    # CHECK:               %[[QK_REG_0:.+]] = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [8], strides = [1]}
    # CHECK:               %[[QK_REG_1:.+]] = vector.extract_strided_slice {{.*}} {offsets = [8], sizes = [8], strides = [1]}
    # CHECK:                amdgpu.mfma %[[QK_REG_0]] * %[[V_REG_F8_0]]{{.*}} {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
    # CHECK:                amdgpu.mfma %[[QK_REG_1]] * %[[V_REG_F8_1]]{{.*}} {blocks = 1 : i32, k = 16 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
    # CHECK:             scf.yield


@run_test
def test_chained_gemm_16x16x32():
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    BLOCK_K2 = tkl.sym.BLOCK_K2

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = [tkw.MMAType.F32_16x16x32_F8, tkw.MMAType.F32_16x16x32_K4_F8]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant[0],
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def chained_gemm_16x16x32(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K2, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            inner_acc = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            q_reg = tkw.cast(q_reg, tkl.f8e4m3fnuz)
            k_reg = tkw.cast(k_reg, tkl.f8e4m3fnuz)
            kq_reg = tkw.mma(k_reg, q_reg, inner_acc)
            qk_reg = tkw.permute(kq_reg, target_shape=[B, M, K2])
            qk_cast_reg = tkw.cast(qk_reg, tkl.f8e4m3fnuz)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            v_reg = tkw.cast(v_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(qk_cast_reg, v_reg, acc, mfma_variant[1])
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K1: 32,
            K2: 256,
            B: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant[0]),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    chained_gemm_16x16x32 = wave_compile(options, chained_gemm_16x16x32)
    print(chained_gemm_16x16x32.asm)

    # CHECK-LABEL:     func.func @chained_gemm_16x16x32(
    # CHECK:             %[[V_SHARED:.+]] = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
    # CHECK:             {{.*}} = scf.for

    # Loading V from shared memory with interleaved/k-width=4, then using insert slice to combine them together.
    # This is to align V's layout with the layout of 1st MMA output.
    # CHECK-COUNT-2:       vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
    # CHECK-COUNT-2:       vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
    # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
    # CHECK-COUNT-2:       arith.truncf {{.*}} : vector<8xf16> to vector<8xf8E4M3FNUZ>

    # 2nd MMA
    # CHECK:               {{.*}} = vector.insert_strided_slice {{.*}} {offsets = [0], strides = [1]}
    # CHECK:               {{.*}} = vector.insert_strided_slice {{.*}} {offsets = [4], strides = [1]}
    # CHECK:               {{.*}} = vector.insert_strided_slice {{.*}} {offsets = [0], strides = [1]}
    # CHECK:               {{.*}} = vector.insert_strided_slice {{.*}} {offsets = [4], strides = [1]}
    # CHECK-COUNT-4:       {{.*}} = amdgpu.mfma
    # CHECK:             scf.yield


@run_test
def test_gemm_pipelined():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_pipelined(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MODULO,
        use_scheduling_barriers=True,
        compile_to_mlir=True,
    )

    gemm_pipelined = wave_compile(options, gemm_pipelined)
    print(gemm_pipelined.asm)

    # CHECK-LABEL:    func.func @gemm_pipelined
    # CHECK-COUNT-2:    vector.load
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-10:   vector.load
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-1:    scf.for
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-6:    vector.load
    # CHECK-COUNT-3:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-4:    vector.load
    # CHECK-COUNT-1:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-1:    scf.yield
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-8:    vector.load
    # CHECK-COUNT-8:    amdgpu.mfma


@run_test
def test_gemm_prefetch():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_prefetch(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 128,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_scheduling_barriers=True,
        compile_to_mlir=True,
    )

    gemm_prefetch = wave_compile(options, gemm_prefetch)
    print(gemm_prefetch.asm)
    # CHECK-LABEL:    func.func @gemm_prefetch
    # Prologue
    # CHECK-COUNT-2:  vector.load
    # CHECK-COUNT-2:  vector.store

    # Steady State
    # CHECK:          scf.for
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # Steady State Local Read
    # CHECK-COUNT-4:    vector.load %alloc
    # CHECK-COUNT-4:    vector.load %alloc_0

    # Steady State Global Read
    # CHECK-COUNT-2:    vector.load {{.*}} : memref<128x128xf16, strided<[128, 1], offset: ?>>, vector<8xf16>
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"

    # Compute
    # CHECK-COUNT-8:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier

    # Steady State Local Write
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK:          scf.yield

    # Prologue
    # CHECK-COUNT-4:  vector.load %alloc
    # CHECK-COUNT-4:  vector.load %alloc_0
    # CHECK-COUNT-8:  amdgpu.mfma


@run_test
def test_dynamic_gemm_pipelined():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    constraints += [tkw.Assumption(K > 4 * BLOCK_K)]

    @tkw.wave(constraints)
    def dynamic_gemm_pipelined(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MODULO,
        use_scheduling_barriers=True,
        dynamic_symbols=(M, N, K),
        dynamic_symbols_map={M: 64, N: 128, K: 256},
        compile_to_mlir=True,
    )
    dynamic_gemm_pipelined = wave_compile(options, dynamic_gemm_pipelined)
    print(dynamic_gemm_pipelined.asm)

    # CHECK-LABEL:    func.func @dynamic_gemm_pipelined
    # CHECK-COUNT-2:    vector.maskedload
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-4:    vector.load
    # CHECK-COUNT-2:    vector.maskedload
    # CHECK-COUNT-4:    vector.load
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-1:    scf.for
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-4:    vector.load
    # CHECK-COUNT-2:    vector.maskedload
    # CHECK-COUNT-3:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-4:    vector.load
    # CHECK-COUNT-1:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-2:    vector.store
    # CHECK-COUNT-2:    llvm.call_intrinsic "llvm.amdgcn.sched.group.barrier"
    # CHECK-COUNT-1:    scf.yield
    # CHECK-COUNT-4:    amdgpu.mfma
    # CHECK-COUNT-1:    amdgpu.lds_barrier
    # CHECK-COUNT-8:    vector.load
    # CHECK-COUNT-8:    amdgpu.mfma


# This test that our stack is able to handle MMA layout with interleaved VGPR offsets/chunks

# e.g a vector<16xf16> may be owned by lane 0, and lane 16 in this layout:
# [0, 0, 0, 0, 16, 16, 16, 16, 0, 0, 0, 0, 16, 16, 16, 16].
# To the lane it should just look like vector<8xf16>.
# Hence for this example, we'd need two reads of vector<4xf16> and insert_slices to
# combine it to a single vector<8xf16>.


@run_test
def test_gemm_with_gpr_offsets():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x32_K4_F8,
        )
    ]

    @tkw.wave(constraints)
    def gemm_with_interleave_gpr(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            a_reg = tkw.cast(a_reg, tkl.f8e4m3fnuz)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.cast(b_reg, tkl.f8e4m3fnuz)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    gemm_with_interleave_gpr = wave_compile(options, gemm_with_interleave_gpr)
    print(gemm_with_interleave_gpr.asm)

    # CHECK-LABEL:    test_gemm_with_gpr_offsets
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 4 + 16)>
    # CHECK:          func.func @gemm_with_interleave_gpr
    # CHECK:            %[[thread_id_x:.*]] = gpu.thread_id  x
    # CHECK:            %[[GPR_OFFSET_0:.+]] = affine.apply #[[MAP0]]()[%[[thread_id_x]]]
    # CHECK:            %[[GPR_OFFSET_1:.+]] = affine.apply #[[MAP1]]()[%[[thread_id_x]]]

    # CHECK:            %[[RHS_0:.+]] = vector.load %alloc[%{{.*}}, %[[GPR_OFFSET_0]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            %[[RHS_1:.+]] = vector.load %alloc[%{{.*}}, %[[GPR_OFFSET_1]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            %[[RHS_INSERT_0:.+]] = vector.insert_strided_slice %[[RHS_0]], %cst {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
    # CHECK:            %[[RHS:.+]] = vector.insert_strided_slice %[[RHS_1]], %[[RHS_INSERT_0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>

    # CHECK:            %[[LHS_0:.+]] = vector.load %alloc_1[%{{.*}}, %[[GPR_OFFSET_0]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            %[[LHS_1:.+]] = vector.load %alloc_1[%{{.*}}, %[[GPR_OFFSET_1]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK:            %[[LHS_INSERT_0:.+]] = vector.insert_strided_slice %[[LHS_0]], %cst {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
    # CHECK:            %[[LHS:.+]] = vector.insert_strided_slice %[[LHS_1]], %[[LHS_INSERT_0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
    # CHECK:            %[[LHS_F8:.+]] = arith.truncf %[[LHS]] : vector<8xf16> to vector<8xf8E4M3FNUZ>
    # CHECK:            %[[RHS_F8:.+]] = arith.truncf %[[RHS]] : vector<8xf16> to vector<8xf8E4M3FNUZ>
    # CHECK:            amdgpu.mfma %[[LHS_F8]] * %[[RHS_F8]]


# This test is used to check three things
# 1. Reduction with multiple different types(MMA, ReduceOp) of iterArg works
# 2. ReduceOp lowering works using constraints from MMA (not just vector_shape).
# 3. We can propagate layout of multiple Reduction results through IterArg/GetResult
#    and observe that broadcast is being generated to resolve binaryOp.
@run_test
def test_gemm_and_reduce():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x16_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        init_max = tkl.Register[M, tkl.f16](-1e6)

        @tkw.reduction(K, init_args=[init_max, c_reg])
        def repeat(
            partial_max: tkl.Register[M, tkl.f16], acc: tkl.Register[M, N, tkl.f32]
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            partial_max = tkw.max(a_reg, partial_max, dim=K)
            acc = tkw.mma(a_reg, b_reg, acc)
            return partial_max, acc

        res_max, res_mm = repeat
        res = res_mm / tkw.cast(res_max, tkl.f32)
        tkw.write(res, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    gemm = wave_compile(options, gemm)
    print(gemm.asm)

    # CHECK-LABEL: test_gemm_and_reduce
    # CHECK:       func.func @gemm
    # CHECK-DAG: %[[C0_IDX:.+]] = arith.constant 0 : index
    # CHECK-DAG: %[[C4_IDX:.+]] = arith.constant 4 : index
    # CHECK-DAG: %[[C1_IDX:.+]] = arith.constant 1 : index

    # Tile Reduction Loop
    # Note: Shape is 32x20 instead of 32x16 because of padding to avoid bank conflicts
    # CHECK: %[[LOOP:.+]]:2 = scf.for %[[ITER:.+]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C1_IDX]]
    # CHECK-SAME: iter_args(%[[ACC0:.+]] = %{{.*}}, %[[ACC1:.+]] = {{.*}})
    # CHECK-COUNT-2: vector.load{{.*}} memref<32x20xf16, #gpu.address_space<workgroup>>, vector<4xf16>
    # CHECK-COUNT-2: gpu.shuffle  xor
    #         CHECK: %[[MAX:.+]] = arith.maximumf %[[ACC0]], %{{.*}}
    #         CHECK: %[[MMA:.+]] = amdgpu.mfma %{{.*}} * %{{.*}} + %[[ACC1]]
    #         CHECK: scf.yield %[[MAX]], %[[MMA]] : vector<1xf16>, vector<4xf32>
    # CHECK: %[[MAX_EXT:.+]] = arith.extf %[[LOOP]]#0 : vector<1xf16> to vector<1xf32>
    # CHECK: %[[BCAST_SRC:.+]] = vector.extract %[[MAX_EXT]][0] : f32 from vector<1xf32>
    # CHECK: %[[BROADCAST:.+]] = vector.splat %[[BCAST_SRC]] : vector<4xf32>
    # CHECK: arith.divf %[[LOOP]]#1, %[[BROADCAST]] : vector<4xf32>


# This test that our stack is able to handle VMMA layout with maximized width read in the K-dimension.
# Things of significance to look out here is for:
# 1. Reads from shared to register are in 8xf16 instead of 4xf16 (typical of native MMA layout).
# 2. We use extract_strided_slice to break "coalesced loads" into 2 reads.
# 3. We generate 2 MFMA that takes in each of the "broken apart" reads, and 1st MMA feed into 2nd MMA.
# 4. The actual MMA uses the native MMA size the VMMA is based on (F16_32x32x8_F16 for this case).


@run_test
def test_gemm_with_maximized_shared_read_32x32x16():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_32x32x16_K8_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_with_maximized_shared_read_32x32x16(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 128,
            N: 128,
            K: 64,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 16,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    gemm_with_maximized_shared_read_32x32x16 = wave_compile(
        options, gemm_with_maximized_shared_read_32x32x16
    )
    print(gemm_with_maximized_shared_read_32x32x16.asm)

    # CHECK-LABEL:    func.func @gemm_with_maximized_shared_read_32x32x16
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
    # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>
    # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>

    # CHECK:            %[[RHS_SHARED_READ:.+]] = vector.load %[[ALLOC]][{{.+}}] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    # CHECK:            %[[LHS_SHARED_READ:.+]] = vector.load %[[ALLOC_0]][{{.+}}] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    # CHECK:            %[[LHS_SLICE_0:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[RHS_SLICE_0:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[MMA_SLICE_0:.+]] = amdgpu.mfma %[[LHS_SLICE_0]] * %[[RHS_SLICE_0]] + %{{..+}} {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>

    # CHECK:            %[[LHS_SLICE_1:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[RHS_SLICE_1:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[MMA_SLICE_1:.+]] = amdgpu.mfma %[[LHS_SLICE_1]] * %[[RHS_SLICE_1]] + %[[MMA_SLICE_0]] {blocks = 1 : i32, k = 8 : i32, m = 32 : i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
    # CHECK:            scf.yield %[[MMA_SLICE_1]] : vector<16xf32>


# This test that our stack is able to handle VMMA layout with maximized width read in the K-dimension.
# Things of significance to look out here is for:
# 1. Reads from shared to register are in 8xf16 instead of 4xf16 (typical of native MMA layout).
# 2. We use extract_strided_slice to break "coalesced loads" into 2 reads.
# 3. We generate 2 MFMA that takes in each of the "broken apart" reads, and 1st MMA feed into 2nd MMA.
# 4. The actual MMA uses the native MMA size the VMMA is based on (F16_16x16x16_F16 for this case).


@run_test
def test_gemm_with_maximized_shared_read_16x16x32():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x32_K8_F16,
        )
    ]

    @tkw.wave(constraints)
    def gemm_with_maximized_shared_read_16x16x32(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            M: 64,
            N: 64,
            K: 128,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )

    gemm_with_maximized_shared_read_16x16x32 = wave_compile(
        options, gemm_with_maximized_shared_read_16x16x32
    )
    print(gemm_with_maximized_shared_read_16x16x32.asm)

    # CHECK-LABEL:    func.func @gemm_with_maximized_shared_read_16x16x32
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
    # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x36xf16, #gpu.address_space<workgroup>>
    # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x36xf16, #gpu.address_space<workgroup>>

    # CHECK:            %[[RHS_SHARED_READ:.+]] = vector.load %[[ALLOC]][{{.+}}] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    # CHECK:            %[[LHS_SHARED_READ:.+]] = vector.load %[[ALLOC_0]][{{.+}}] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    # CHECK:            %[[LHS_SLICE_0:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[RHS_SLICE_0:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[MMA_SLICE_0:.+]] = amdgpu.mfma %[[LHS_SLICE_0]] * %[[RHS_SLICE_0]] + %{{..+}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

    # CHECK:            %[[LHS_SLICE_1:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[RHS_SLICE_1:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
    # CHECK:            %[[MMA_SLICE_1:.+]] = amdgpu.mfma %[[LHS_SLICE_1]] * %[[RHS_SLICE_1]] + %[[MMA_SLICE_0]] {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    # CHECK:            scf.yield %[[MMA_SLICE_1]] : vector<4xf32>


@run_test
def test_broadcast_batched_gemm_with_vmma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(B_KV, BLOCK_B, 2, primary=False)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x32_K8_F16,
            vector_shapes={B: 0, B_KV: 0, M: 16, N: 16},
        )
    ]

    head_ratio = B // B_KV
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    a_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B_KV: i // head_ratio, M: j, K: k},
        outputs={B_KV: i, M: j, K: k},
    )

    @tkw.wave(constraints)
    def broadcast_batched_gemm_with_vmma(
        a: tkl.Memory[B_KV, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[B, N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(
                a, elements_per_thread=LOAD_ELEMS_PER_THREAD, mapping=a_mapping
            )
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    options = WaveCompileOptions(
        subs={
            B: 6,
            B_KV: 1,
            M: 64,
            N: 128,
            K: 64,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            BLOCK_B: 1,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        compile_to_mlir=True,
    )
    broadcast_batched_gemm_with_vmma = wave_compile(
        options, broadcast_batched_gemm_with_vmma
    )
    print(broadcast_batched_gemm_with_vmma.asm)

    # CHECK-LABEL:    test_broadcast_batched_gemm_with_vmma
    # CHECK-DAG:        #[[MAP:.*]] = affine_map<()[s0] -> (s0 floordiv 6)>
    # CHECK:          func.func @broadcast_batched_gemm_with_vmma
    # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
    # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
    # CHECK:            %[[WG_ID2:.+]] = stream.dispatch.workgroup.id[2] : index
    # CHECK:            %[[RHS_GLOBAL:.+]] = stream.binding.subspan %[[ARG1]]
    # CHECK:            %[[LHS_GLOBAL:.+]] = stream.binding.subspan %[[ARG0]]
    # CHECK:            %[[HKV_IDX:.+]] = affine.apply #[[MAP]]()[%[[WG_ID2]]]
    # CHECK:            scf.for
    # CHECK:             %[[LHS_READ:.+]] = vector.load %[[LHS_GLOBAL]][%[[HKV_IDX]], %{{.+}}, {{.+}}] : {{.*}}, vector<8xf16>
    # CHECK:             %[[RHS_READ:.+]] = vector.load %[[RHS_GLOBAL]][%[[WG_ID2]], %{{.+}}, {{.+}}] : {{.*}}, vector<8xf16>
    # CHECK-COUNT-2:     vector.extract_strided_slice
    # CHECK-COUNT-1:     amdgpu.mfma
    # CHECK-COUNT-2:     vector.extract_strided_slice
    # CHECK-COUNT-1:     amdgpu.mfma
    # CHECK:            scf.yield
