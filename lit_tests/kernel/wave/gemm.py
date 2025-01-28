# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
    get_default_compile_config,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
import torch

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


def codegen_test_context(canonicalize: bool = False, dynamic_symbols=[]):
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

    return tk.gen.TestLaunchContext(
        bindings, canonicalize=canonicalize, dynamic_symbols=dynamic_symbols
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm(a, b, c).module_op)

        # CHECK-LABEL:    func.func @gemm
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x20xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x20xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<64x64xf16,
        # CHECK-SAME:         strided<[64, 1], offset: ?>>
        # CHECK:            %[[D1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<128x64xf16,
        # CHECK-SAME:         strided<[64, 1], offset: ?>>
        # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D4:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D5:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D6:.+]] = arith.addi %[[D5]], %[[D4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D7:.+]] = arith.addi %[[D6]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D8:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D9:.+]] = arith.divsi %[[D8]], %[[C16]] : index
        # CHECK:            %[[D10:.+]] = arith.muli %[[D9]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D11:.+]] = arith.addi %[[D5]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D12:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D13:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D14:.+]] = arith.addi %[[D5]], %[[D13]] overflow<nsw, nuw> : index
        # CHECK:            %[[D15:.+]] = arith.addi %[[D14]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D16:.+]] = arith.addi %[[D5]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D17:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
        # CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<4xf32>) {
        # CHECK:              %[[D39:.+]] = arith.muli %[[ARG3]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:              %[[D40:.+]] = arith.addi %[[D39]], %[[D10]] overflow<nsw, nuw> : index
        # CHECK:              %[[D41:.+]] = vector.load %[[D0]][%[[D7]], %[[D40]]] : memref<64x64xf16, strided<[64, 1], offset:
        # CHECK-SAME:           ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D41]], %[[ALLOC]][%[[D11]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D42:.+]] = vector.load %[[ALLOC]][%[[D11]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D43:.+]] = vector.load %[[D1]][%[[D15]], %[[D40]]] : memref<128x64xf16, strided<[64, 1],
        # CHECK-SAME:           offset: ?>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              vector.store %[[D43]], %[[ALLOC_0]][%[[D16]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D44:.+]] = vector.load %[[ALLOC_0]][%[[D16]], %[[D10]]] : memref<32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D45:.+]] = amdgpu.mfma %[[D42]] * %[[D44]] + %[[ARG4]] {blocks = 1 : i32, k = 16 : i32, m = 16
        # CHECK-SAME:           : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:              scf.yield %[[D45]] : vector<4xf32>
        # CHECK:            }
        # CHECK:            %[[D18:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D19:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<64x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>
        # CHECK:            %[[D20:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D21:.+]] = arith.divsi %[[D20]], %[[C16]] : index
        # CHECK:            %[[D22:.+]] = arith.muli %[[D21]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D23:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D24:.+]] = arith.muli %[[D23]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D25:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D26:.+]] = arith.addi %[[D25]], %[[D24]] overflow<nsw, nuw> : index
        # CHECK:            %[[D27:.+]] = arith.addi %[[D26]], %[[D22]] overflow<nsw, nuw> : index
        # CHECK:            %[[D28:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D29:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D30:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D31:.+]] = arith.addi %[[D30]], %[[D29]] overflow<nsw, nuw> : index
        # CHECK:            %[[D32:.+]] = arith.addi %[[D31]], %[[D28]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D18]], %[[D19]][%[[D27]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D33:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D34:.+]] = arith.addi %[[D27]], %[[C1]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D33]], %[[D19]][%[[D34]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D35:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D36:.+]] = arith.addi %[[D27]], %[[C2]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D35]], %[[D19]][%[[D36]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D37:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D38:.+]] = arith.addi %[[D27]], %[[C3]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D37]], %[[D19]][%[[D38]], %[[D32]]] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            return


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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.ones(64, 32, dtype=torch.int8)
        b = torch.ones(128, 32, dtype=torch.int8)
        c = torch.zeros(64, 128, dtype=torch.int32)
        print(cdna2_int_gemm(a, b, c).module_op)

        # CHECK-LABEL:    func.func @cdna2_int_gemm
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0> : vector<4xi32>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x24xi8, #gpu.address_space<workgroup>>
        # CHECK:            %[[ALLOC_1:.+]] = memref.alloc() : memref<32x24xi8, #gpu.address_space<workgroup>>
        # CHECK:            %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK:            %[[GLOBAL_1:.+]] = stream.binding.subspan %[[ARG1]]
        # CHECK:            scf.for %[[IVAR:.+]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[CST]]) -> (vector<4xi32>) {
        # CHECK:                %[[REG_0:.+]] = vector.load %[[GLOBAL_0]]
        # CHECK:                vector.store %[[REG_0]], %[[ALLOC_0]]
        # CHECK:                %[[LHS:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<32x24xi8, #gpu.address_space<workgroup>>, vector<4xi8>
        # CHECK:                %[[REG_1:.+]] = vector.load %[[GLOBAL_1]]
        # CHECK:                vector.store %[[REG_1]], %[[ALLOC_1]]
        # CHECK:                %[[RHS:.+]] = vector.load %[[ALLOC_1]]{{.*}} : memref<32x24xi8, #gpu.address_space<workgroup>>, vector<4xi8>
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.ones(64, 32, dtype=torch.int8)
        b = torch.ones(128, 32, dtype=torch.int8)
        c = torch.zeros(64, 128, dtype=torch.int32)
        print(cdna3_int_gemm(a, b, c).module_op)

        # CHECK-LABEL:    func.func @cdna3_int_gemm
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0> : vector<4xi32>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x40xi8, #gpu.address_space<workgroup>>
        # CHECK:            %[[ALLOC_1:.+]] = memref.alloc() : memref<32x40xi8, #gpu.address_space<workgroup>>
        # CHECK:            %[[GLOBAL_0:.+]] = stream.binding.subspan %[[ARG0]]
        # CHECK:            %[[GLOBAL_1:.+]] = stream.binding.subspan %[[ARG1]]
        # CHECK:            scf.for %[[IVAR:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[CST]]) -> (vector<4xi32>) {
        # CHECK:                %[[REG_0:.+]] = vector.load %[[GLOBAL_0]]
        # CHECK:                vector.store %[[REG_0]], %[[ALLOC_0]]
        # CHECK:                %[[LHS:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<32x40xi8, #gpu.address_space<workgroup>>, vector<8xi8>
        # CHECK:                %[[REG_1:.+]] = vector.load %[[GLOBAL_1]]
        # CHECK:                vector.store %[[REG_1]], %[[ALLOC_1]]
        # CHECK:                %[[RHS:.+]] = vector.load %[[ALLOC_1]]{{.*}} : memref<32x40xi8, #gpu.address_space<workgroup>>, vector<8xi8>
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
            acc: tkl.Register[B, M, N, tkl.f32]
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(12, 64, 32, dtype=torch.float16)
        b = torch.randn(12, 128, 32, dtype=torch.float16)
        c = torch.zeros(12, 64, 128, dtype=torch.float32)
        print(batched_gemm(a, b, c).module_op)

        # CHECK-LABEL:    func.func @batched_gemm
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]:
        # CHECK-SAME:       !stream.binding, %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info =
        # CHECK-SAME:       #[[TRANSLATION:.+]]} {
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C32:.+]] = arith.constant 32 : index
        # CHECK-DAG:        %[[C64:.+]] = arith.constant 64 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C4:.+]] = arith.constant 4 : index
        # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK:            %[[WORKGROUP_ID_0:.+]] = stream.dispatch.workgroup.id[0] : index
        # CHECK:            %[[WORKGROUP_ID_1:.+]] = stream.dispatch.workgroup.id[1] : index
        # CHECK:            %[[WORKGROUP_ID_2:.+]] = stream.dispatch.workgroup.id[2] : index
        # CHECK-DAG:        %[[THREAD_ID_X:.+]] = gpu.thread_id  x
        # CHECK-DAG:        %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<1x32x20xf16, #[[GPU:.+]].address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<1x32x20xf16, #[[GPU]].address_space<workgroup>>
        # CHECK:            %[[D0:.+]] = stream.binding.subspan %[[ARG0]][%[[C0]]] : !stream.binding -> memref<12x64x64xf16,
        # CHECK-SAME:         strided<[4096, 64, 1], offset: ?>>
        # CHECK:            %[[D1:.+]] = stream.binding.subspan %[[ARG1]][%[[C0]]] : !stream.binding -> memref<12x128x64xf16,
        # CHECK-SAME:         strided<[8192, 64, 1], offset: ?>>
        # CHECK:            %[[D2:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D3:.+]] = arith.muli %[[D2]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D4:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D5:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D6:.+]] = arith.addi %[[D5]], %[[D4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D7:.+]] = arith.addi %[[D6]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D8:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D9:.+]] = arith.divsi %[[D8]], %[[C16]] : index
        # CHECK:            %[[D10:.+]] = arith.muli %[[D9]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D11:.+]] = arith.addi %[[D5]], %[[D3]] overflow<nsw, nuw> : index
        # CHECK:            %[[D12:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D13:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D14:.+]] = arith.addi %[[D5]], %[[D13]] overflow<nsw, nuw> : index
        # CHECK:            %[[D15:.+]] = arith.addi %[[D14]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D16:.+]] = arith.addi %[[D5]], %[[D12]] overflow<nsw, nuw> : index
        # CHECK:            %[[D17:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
        # CHECK-SAME:         iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<4xf32>) {
        # CHECK:              %[[D39:.+]] = arith.muli %[[ARG3]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:              %[[D40:.+]] = arith.addi %[[D39]], %[[D10]] overflow<nsw, nuw> : index
        # CHECK:              %[[D41:.+]] = vector.load %[[D0]][%[[WORKGROUP_ID_2]], %[[D7]], %[[D40]]] : memref<12x64x64xf16,
        # CHECK-SAME:           strided<[4096, 64, 1], offset: ?>>, vector<4xf16>
        # CHECK:              vector.store %[[D41]], %[[ALLOC]][%[[C0]], %[[D11]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D42:.+]] = vector.load %[[ALLOC]][%[[C0]], %[[D11]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D43:.+]] = vector.load %[[D1]][%[[WORKGROUP_ID_2]], %[[D15]], %[[D40]]] : memref<12x128x64xf16,
        # CHECK-SAME:           strided<[8192, 64, 1], offset: ?>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              vector.store %[[D43]], %[[ALLOC_0]][%[[C0]], %[[D16]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              amdgpu.lds_barrier
        # CHECK:              %[[D44:.+]] = vector.load %[[ALLOC_0]][%[[C0]], %[[D16]], %[[D10]]] : memref<1x32x20xf16,
        # CHECK-SAME:           #[[GPU]].address_space<workgroup>>, vector<4xf16>
        # CHECK:              %[[D45:.+]] = amdgpu.mfma %[[D42]] * %[[D44]] + %[[ARG4]] {blocks = 1 : i32, k = 16 : i32, m = 16
        # CHECK-SAME:           : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:              scf.yield %[[D45]] : vector<4xf32>
        # CHECK:            }
        # CHECK:            %[[D18:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D19:.+]] = stream.binding.subspan %[[ARG2]][%[[C0]]] : !stream.binding -> memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>
        # CHECK:            %[[D20:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D21:.+]] = arith.divsi %[[D20]], %[[C16]] : index
        # CHECK:            %[[D22:.+]] = arith.muli %[[D21]], %[[C4]] overflow<nsw, nuw> : index
        # CHECK:            %[[D23:.+]] = arith.divsi %[[THREAD_ID_X]], %[[C64]] : index
        # CHECK:            %[[D24:.+]] = arith.muli %[[D23]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D25:.+]] = arith.muli %[[WORKGROUP_ID_0]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D26:.+]] = arith.addi %[[D25]], %[[D24]] overflow<nsw, nuw> : index
        # CHECK:            %[[D27:.+]] = arith.addi %[[D26]], %[[D22]] overflow<nsw, nuw> : index
        # CHECK:            %[[D28:.+]] = arith.muli %[[THREAD_ID_Y]], %[[C16]] overflow<nsw, nuw> : index
        # CHECK:            %[[D29:.+]] = arith.muli %[[WORKGROUP_ID_1]], %[[C32]] overflow<nsw, nuw> : index
        # CHECK:            %[[D30:.+]] = arith.remsi %[[THREAD_ID_X]], %[[C16]] : index
        # CHECK:            %[[D31:.+]] = arith.addi %[[D30]], %[[D29]] overflow<nsw, nuw> : index
        # CHECK:            %[[D32:.+]] = arith.addi %[[D31]], %[[D28]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D18]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D27]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D33:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D34:.+]] = arith.addi %[[D27]], %[[C1]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D33]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D34]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D35:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D36:.+]] = arith.addi %[[D27]], %[[C2]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D35]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D36]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            %[[D37:.+]] = vector.extract_strided_slice %[[D17]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D38:.+]] = arith.addi %[[D27]], %[[C3]] overflow<nsw, nuw> : index
        # CHECK:            vector.store %[[D37]], %[[D19]][%[[WORKGROUP_ID_2]], %[[D38]], %[[D32]]] : memref<12x64x128xf32,
        # CHECK-SAME:         strided<[8192, 128, 1], offset: ?>>, vector<1xf32>
        # CHECK:            return


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
            acc: tkl.Register[B, M, N, tkl.f32]
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        q = torch.randn(8, 64, 64, dtype=torch.float16)
        k = torch.randn(8, 256, 64, dtype=torch.float16)
        v = torch.zeros(8, 128, 256, dtype=torch.float16)
        output = torch.zeros(8, 64, 128, dtype=torch.float32)
        print(chained_gemm(q, k, v, output).module_op)

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
            acc: tkl.Register[B, M, N, tkl.f32]
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        q = torch.randn(8, 64, 64, dtype=torch.float16)
        k = torch.randn(8, 256, 64, dtype=torch.float16)
        v = torch.zeros(8, 128, 256, dtype=torch.float16)
        output = torch.zeros(8, 64, 128, dtype=torch.float32)
        print(chained_gemm_32x32x8(q, k, v, output).module_op)

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
            acc: tkl.Register[B, M, N, tkl.f32]
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        q = torch.randn(8, 64, 64, dtype=torch.float16)
        k = torch.randn(8, 256, 64, dtype=torch.float16)
        v = torch.zeros(8, 128, 256, dtype=torch.float16)
        output = torch.zeros(8, 64, 128, dtype=torch.float32)
        print(chained_gemm_32x32x16(q, k, v, output).module_op)

        # CHECK-LABEL:     func.func @chained_gemm_32x32x16(
        # CHECK:             %[[V_SHARED:.+]] = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
        # CHECK:             {{.*}} = scf.for
        # 1st MMA
        # CHECK-COUNT-4:       {{.*}} = arith.truncf
        # CHECK-COUNT-2:       {{.*}} = amdgpu.mfma

        # Loading V from shared memory with interleaved/k-width=4, then using insert slice to combine them together.
        # This is to align V's layout with the layout of 1st MMA output.
        # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
        # CHECK-COUNT-2:       %[[V_REG_0:.+]] = vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
        # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
        # CHECK-COUNT-2:       %[[V_REG_1:.+]] = vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
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
            acc: tkl.Register[B, M, N, tkl.f32]
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        q = torch.randn(8, 64, 64, dtype=torch.float16)
        k = torch.randn(8, 256, 64, dtype=torch.float16)
        v = torch.zeros(8, 128, 256, dtype=torch.float16)
        output = torch.zeros(8, 64, 128, dtype=torch.float32)
        print(chained_gemm_16x16x32(q, k, v, output).module_op)

        # CHECK-LABEL:     func.func @chained_gemm_16x16x32(
        # CHECK:             %[[V_SHARED:.+]] = memref.alloc() : memref<1x64x36xf16, #gpu.address_space<workgroup>>
        # CHECK:             {{.*}} = scf.for
        # 1st MMA
        # CHECK-COUNT-4:       {{.*}} = arith.truncf
        # CHECK-COUNT-4:       {{.*}} = amdgpu.mfma

        # Loading V from shared memory with interleaved/k-width=4, then using insert slice to combine them together.
        # This is to align V's layout with the layout of 1st MMA output.
        # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
        # CHECK-COUNT-2:       vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
        # CHECK-COUNT-2:       vector.load %[[V_SHARED]]
        # CHECK-COUNT-2:       vector.insert_strided_slice {{.*}} : vector<4xf16> into vector<8xf16>
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

    with tk.gen.TestLaunchContext(
        {
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
        schedule=True,
        use_scheduling_barriers=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm_pipelined(a, b, c).module_op)

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

    with tk.gen.TestLaunchContext(
        {
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
        schedule=True,
        use_scheduling_barriers=True,
        dynamic_symbols=(M, N, K),
        dynamic_symbols_map={M: 64, N: 128, K: 256},
    ):
        a = torch.randn(64, 256, dtype=torch.float16)
        b = torch.randn(128, 256, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(dynamic_gemm_pipelined(a, b, c).module_op)

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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(64, 64, dtype=torch.float16)
        b = torch.randn(64, 64, dtype=torch.float16)
        c = torch.zeros(64, 64, dtype=torch.float32)
        print(gemm_with_interleave_gpr(a, b, c).module_op)

        # CHECK-LABEL:    func.func @gemm_with_interleave_gpr
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK:            %[[LANE_ID:.+]] = arith.remsi %thread_id_x, %c64 : index
        # CHECK:            %[[D0:.+]] = arith.divsi %[[LANE_ID]], %c16 : index
        # CHECK:            %[[GPR_OFFSET_0:.+]] = arith.muli %[[D0]], %c4
        # CHECK:            %[[GPR_OFFSET_1:.+]] = arith.addi %[[GPR_OFFSET_0]], %c16

        # CHECK:            %[[LHS_0:.+]] = vector.load %alloc[%{{.*}}, %[[GPR_OFFSET_0]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[LHS_1:.+]] = vector.load %alloc[%{{.*}}, %[[GPR_OFFSET_1]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[LHS_INSERT_0:.+]] = vector.insert_strided_slice %[[LHS_0]], %cst {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
        # CHECK:            %[[LHS:.+]] = vector.insert_strided_slice %[[LHS_1]], %[[LHS_INSERT_0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
        # CHECK:            %[[LHS_F8:.+]] = arith.truncf %[[LHS]] : vector<8xf16> to vector<8xf8E4M3FNUZ>

        # CHECK:            %[[RHS_0:.+]] = vector.load %alloc_1[%{{.*}}, %[[GPR_OFFSET_0]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[RHS_1:.+]] = vector.load %alloc_1[%{{.*}}, %[[GPR_OFFSET_1]]] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<4xf16>
        # CHECK:            %[[RHS_INSERT_0:.+]] = vector.insert_strided_slice %[[RHS_0]], %cst {offsets = [0], strides = [1]} : vector<4xf16> into vector<8xf16>
        # CHECK:            %[[RHS:.+]] = vector.insert_strided_slice %[[RHS_1]], %[[RHS_INSERT_0]] {offsets = [4], strides = [1]} : vector<4xf16> into vector<8xf16>
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(gemm(a, b, c).module_op)
        # CHECK-LABEL: func.func @gemm
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
        # CHECK: %[[BROADCAST:.+]] = vector.splat %19 : vector<4xf32>
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(128, 64, dtype=torch.float16)
        b = torch.randn(128, 64, dtype=torch.float16)
        c = torch.zeros(128, 128, dtype=torch.float32)
        print(gemm_with_maximized_shared_read_32x32x16(a, b, c).module_op)

        # CHECK-LABEL:    func.func @gemm_with_maximized_shared_read_32x32x16
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<64x20xf16, #gpu.address_space<workgroup>>

        # CHECK:            %[[LHS_SHARED_READ:.+]] = vector.load %[[ALLOC]][{{.+}}] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        # CHECK:            %[[RHS_SHARED_READ:.+]] = vector.load %[[ALLOC_0]][{{.+}}] : memref<64x20xf16, #gpu.address_space<workgroup>>, vector<8xf16>
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

    with tk.gen.TestLaunchContext(
        {
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
    ):
        a = torch.randn(64, 128, dtype=torch.float16)
        b = torch.randn(64, 128, dtype=torch.float16)
        c = torch.zeros(64, 64, dtype=torch.float32)
        print(gemm_with_maximized_shared_read_16x16x32(a, b, c).module_op)

        # CHECK-LABEL:    func.func @gemm_with_maximized_shared_read_16x16x32
        # CHECK-SAME:       (%[[ARG0:[a-zA-Z0-9_]+]]: !stream.binding, %[[ARG1:[a-zA-Z0-9_]+]]: !stream.binding,
        # CHECK-SAME:       %[[ARG2:[a-zA-Z0-9_]+]]: !stream.binding) attributes {translation_info = #[[TRANSLATION:.+]]} {
        # CHECK:            %[[ALLOC:.+]] = memref.alloc() : memref<32x36xf16, #gpu.address_space<workgroup>>
        # CHECK:            %[[ALLOC_0:.+]] = memref.alloc() : memref<32x36xf16, #gpu.address_space<workgroup>>

        # CHECK:            %[[LHS_SHARED_READ:.+]] = vector.load %[[ALLOC]][{{.+}}] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        # CHECK:            %[[RHS_SHARED_READ:.+]] = vector.load %[[ALLOC_0]][{{.+}}] : memref<32x36xf16, #gpu.address_space<workgroup>>, vector<8xf16>
        # CHECK:            %[[LHS_SLICE_0:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
        # CHECK:            %[[RHS_SLICE_0:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
        # CHECK:            %[[MMA_SLICE_0:.+]] = amdgpu.mfma %[[LHS_SLICE_0]] * %[[RHS_SLICE_0]] + %{{..+}} {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>

        # CHECK:            %[[LHS_SLICE_1:.+]] = vector.extract_strided_slice %[[LHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
        # CHECK:            %[[RHS_SLICE_1:.+]] = vector.extract_strided_slice %[[RHS_SHARED_READ]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
        # CHECK:            %[[MMA_SLICE_1:.+]] = amdgpu.mfma %[[LHS_SLICE_1]] * %[[RHS_SLICE_1]] + %[[MMA_SLICE_0]] {blocks = 1 : i32, k = 16 : i32, m = 16 : i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:            scf.yield %[[MMA_SLICE_1]] : vector<4xf32>
