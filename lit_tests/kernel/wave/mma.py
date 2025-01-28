# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import run_test
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


@run_test
def test_mma():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
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
    def mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 64,
            N: 128,
            K: 16,
            BLOCK_M: 32,
            BLOCK_N: 32,
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
        print(mma(a, b, c).module_op)

        # CHECK:          func.func @mma
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index

        # CHECK-DAG:        %[[ALLOC:.+]] = memref.alloc() : memref<32x20xf16,
        # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.alloc() : memref<32x20xf16,
        # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x20xf16,
        # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x20xf16,
        # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D12]] * %[[D20]] + %[[CST]] {blocks = 1 : i32, k = 16 : i32, m = 16 :
        # CHECK-SAME:         i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
        # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>

        # CHECK-DAG:        vector.store %[[D22]], {{.*}} : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D27:.+]] = arith.addi {{.*}}, %[[C1]]
        # CHECK:            vector.store %[[D26]], {{.*}}[%[[D27]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D29:.+]] = arith.addi {{.*}}, %[[C2]]
        # CHECK:            vector.store %[[D28]], {{.*}}[%[[D29]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D31:.+]] = arith.addi {{.*}}, %[[C3]]
        # CHECK:            vector.store %[[D30]], {{.*}}[%[[D31]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>


@run_test
def test_mma_32x32x8():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_32x32x8_F16,
        )
    ]

    @tkw.wave(constraints)
    def mma_32x32x8(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 128,
            K: 8,
            BLOCK_M: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(mma_32x32x8(a, b, c).module_op)

        # CHECK:          func.func @mma_32x32x8
        # CHECK-DAG:        %[[C27:.+]] = arith.constant 27 : index
        # CHECK-DAG:        %[[C26:.+]] = arith.constant 26 : index
        # CHECK-DAG:        %[[C25:.+]] = arith.constant 25 : index
        # CHECK-DAG:        %[[C24:.+]] = arith.constant 24 : index
        # CHECK-DAG:        %[[C19:.+]] = arith.constant 19 : index
        # CHECK-DAG:        %[[C18:.+]] = arith.constant 18 : index
        # CHECK-DAG:        %[[C17:.+]] = arith.constant 17 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C11:.+]] = arith.constant 11 : index
        # CHECK-DAG:        %[[C10:.+]] = arith.constant 10 : index
        # CHECK-DAG:        %[[C9:.+]] = arith.constant 9 : index
        # CHECK-DAG:        %[[C8:.+]] = arith.constant 8 : index
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index

        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
        # CHECK-DAG:        %[[ALLOC:.+]] = memref.alloc() : memref<64x12xf16,
        # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x12xf16,
        # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.alloc() : memref<64x12xf16,
        # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x12xf16,
        # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D12]] * %[[D20]] + %[[CST]] {blocks = 1 : i32, k = 8 : i32, m = 32 :
        # CHECK-SAME:         i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
        # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D23:.+]] = stream.binding.subspan {{.*}} : !stream.binding -> memref<128x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>

        # CHECK-DAG:        vector.store %[[D22]], %[[D23]][{{.*}}, {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D27:.+]] = arith.addi {{.*}}, %[[C1]]
        # CHECK:            vector.store %[[D26]], %[[D23]][%[[D27]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D29:.+]] = arith.addi {{.*}}, %[[C2]]
        # CHECK:            vector.store %[[D28]], %[[D23]][%[[D29]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D31:.+]] = arith.addi {{.*}}, %[[C3]]
        # CHECK:            vector.store %[[D30]], %[[D23]][%[[D31]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D32:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [4], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D33:.+]] = arith.addi {{.*}}, %[[C8]]
        # CHECK:            vector.store %[[D32]], %[[D23]][%[[D33]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D34:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [5], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D35:.+]] = arith.addi {{.*}}, %[[C9]]
        # CHECK:            vector.store %[[D34]], %[[D23]][%[[D35]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D36:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [6], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D37:.+]] = arith.addi {{.*}}, %[[C10]]
        # CHECK:            vector.store %[[D36]], %[[D23]][%[[D37]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D38:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [7], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D39:.+]] = arith.addi {{.*}}, %[[C11]]
        # CHECK:            vector.store %[[D38]], %[[D23]][%[[D39]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D40:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [8], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D41:.+]] = arith.addi {{.*}}, %[[C16]]
        # CHECK:            vector.store %[[D40]], %[[D23]][%[[D41]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D42:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [9], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D43:.+]] = arith.addi {{.*}}, %[[C17]]
        # CHECK:            vector.store %[[D42]], %[[D23]][%[[D43]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D44:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [10], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D45:.+]] = arith.addi {{.*}}, %[[C18]]
        # CHECK:            vector.store %[[D44]], %[[D23]][%[[D45]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D46:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [11], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D47:.+]] = arith.addi {{.*}}, %[[C19]]
        # CHECK:            vector.store %[[D46]], %[[D23]][%[[D47]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D48:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [12], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D49:.+]] = arith.addi {{.*}}, %[[C24]]
        # CHECK:            vector.store %[[D48]], %[[D23]][%[[D49]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D50:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [13], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D51:.+]] = arith.addi {{.*}}, %[[C25]]
        # CHECK:            vector.store %[[D50]], %[[D23]][%[[D51]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D52:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [14], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D53:.+]] = arith.addi {{.*}}, %[[C26]]
        # CHECK:            vector.store %[[D52]], %[[D23]][%[[D53]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D54:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [15], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D55:.+]] = arith.addi {{.*}}, %[[C27]]
        # CHECK:            vector.store %[[D54]], %[[D23]][%[[D55]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>


@run_test
def test_mma_32x32x16():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_32x32x16_F8,
        )
    ]

    @tkw.wave(constraints)
    def mma_32x32x16(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 128,
            K: 16,
            BLOCK_M: 64,
            BLOCK_N: 64,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 16,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(mma_32x32x16(a, b, c).module_op)

        # CHECK:          func.func @mma_32x32x16
        # CHECK-DAG:        %[[C27:.+]] = arith.constant 27 : index
        # CHECK-DAG:        %[[C26:.+]] = arith.constant 26 : index
        # CHECK-DAG:        %[[C25:.+]] = arith.constant 25 : index
        # CHECK-DAG:        %[[C24:.+]] = arith.constant 24 : index
        # CHECK-DAG:        %[[C19:.+]] = arith.constant 19 : index
        # CHECK-DAG:        %[[C18:.+]] = arith.constant 18 : index
        # CHECK-DAG:        %[[C17:.+]] = arith.constant 17 : index
        # CHECK-DAG:        %[[C16:.+]] = arith.constant 16 : index
        # CHECK-DAG:        %[[C11:.+]] = arith.constant 11 : index
        # CHECK-DAG:        %[[C10:.+]] = arith.constant 10 : index
        # CHECK-DAG:        %[[C9:.+]] = arith.constant 9 : index
        # CHECK-DAG:        %[[C8:.+]] = arith.constant 8 : index
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index

        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
        # CHECK-DAG:        %[[ALLOC:.+]] = memref.alloc() : memref<64x24xf8E4M3FNUZ,
        # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x24xf8E4M3FNUZ,
        # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.alloc() : memref<64x24xf8E4M3FNUZ,
        # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x24xf8E4M3FNUZ,
        # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D12]] * %[[D20]] + %[[CST]] {blocks = 1 : i32, k = 16 : i32, m = 32 :
        # CHECK-SAME:         i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
        # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D23:.+]] = stream.binding.subspan {{.*}} : !stream.binding -> memref<128x128xf32,
        # CHECK-SAME:         strided<[128, 1], offset: ?>>

        # CHECK-DAG:        vector.store %[[D22]], %[[D23]][{{.*}}, {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D27:.+]] = arith.addi {{.*}}, %[[C1]]
        # CHECK:            vector.store %[[D26]], %[[D23]][%[[D27]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D29:.+]] = arith.addi {{.*}}, %[[C2]]
        # CHECK:            vector.store %[[D28]], %[[D23]][%[[D29]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D31:.+]] = arith.addi {{.*}}, %[[C3]]
        # CHECK:            vector.store %[[D30]], %[[D23]][%[[D31]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D32:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [4], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D33:.+]] = arith.addi {{.*}}, %[[C8]]
        # CHECK:            vector.store %[[D32]], %[[D23]][%[[D33]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D34:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [5], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D35:.+]] = arith.addi {{.*}}, %[[C9]]
        # CHECK:            vector.store %[[D34]], %[[D23]][%[[D35]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D36:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [6], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D37:.+]] = arith.addi {{.*}}, %[[C10]]
        # CHECK:            vector.store %[[D36]], %[[D23]][%[[D37]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D38:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [7], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D39:.+]] = arith.addi {{.*}}, %[[C11]]
        # CHECK:            vector.store %[[D38]], %[[D23]][%[[D39]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D40:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [8], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D41:.+]] = arith.addi {{.*}}, %[[C16]]
        # CHECK:            vector.store %[[D40]], %[[D23]][%[[D41]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D42:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [9], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D43:.+]] = arith.addi {{.*}}, %[[C17]]
        # CHECK:            vector.store %[[D42]], %[[D23]][%[[D43]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D44:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [10], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D45:.+]] = arith.addi {{.*}}, %[[C18]]
        # CHECK:            vector.store %[[D44]], %[[D23]][%[[D45]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D46:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [11], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D47:.+]] = arith.addi {{.*}}, %[[C19]]
        # CHECK:            vector.store %[[D46]], %[[D23]][%[[D47]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D48:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [12], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D49:.+]] = arith.addi {{.*}}, %[[C24]]
        # CHECK:            vector.store %[[D48]], %[[D23]][%[[D49]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D50:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [13], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D51:.+]] = arith.addi {{.*}}, %[[C25]]
        # CHECK:            vector.store %[[D50]], %[[D23]][%[[D51]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D52:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [14], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D53:.+]] = arith.addi {{.*}}, %[[C26]]
        # CHECK:            vector.store %[[D52]], %[[D23]][%[[D53]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D54:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [15], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<16xf32> to vector<1xf32>
        # CHECK:            %[[D55:.+]] = arith.addi {{.*}}, %[[C27]]
        # CHECK:            vector.store %[[D54]], %[[D23]][%[[D55]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>


@run_test
def test_mma_16x16x32():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=tkw.MMAType.F32_16x16x32_F8,
        )
    ]

    @tkw.wave(constraints)
    def mma_16x16x32(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(a_reg, b_reg, c_reg)
        tkw.write(acc, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    with tk.gen.TestLaunchContext(
        {
            M: 128,
            N: 128,
            K: 32,
            BLOCK_M: 32,
            BLOCK_N: 32,
            LOAD_ELEMS_PER_THREAD: 8,
            STORE_ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        a = torch.randn(64, 32, dtype=torch.float16)
        b = torch.randn(128, 32, dtype=torch.float16)
        c = torch.zeros(64, 128, dtype=torch.float32)
        print(mma_16x16x32(a, b, c).module_op)

        # CHECK:          func.func @mma_16x16x32
        # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
        # CHECK-DAG:        %[[C1:.+]] = arith.constant 1 : index
        # CHECK-DAG:        %[[C2:.+]] = arith.constant 2 : index
        # CHECK-DAG:        %[[C3:.+]] = arith.constant 3 : index

        # CHECK-DAG:        %[[ALLOC:.+]] = memref.alloc() : memref<32x40xf8E4M3FNUZ,
        # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.alloc() : memref<32x40xf8E4M3FNUZ,
        # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
        # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
        # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D12]] * %[[D20]] + %[[CST]] {blocks = 1 : i32, k = 32 : i32, m = 16 :
        # CHECK-SAME:         i32, n = 16 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
        # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>

        # CHECK-DAG:        vector.store %[[D22]], {{.*}} : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D27:.+]] = arith.addi {{.*}}, %[[C1]]
        # CHECK:            vector.store %[[D26]], {{.*}}[%[[D27]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D29:.+]] = arith.addi {{.*}}, %[[C2]]
        # CHECK:            vector.store %[[D28]], {{.*}}[%[[D29]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
        # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
        # CHECK-SAME:         vector<4xf32> to vector<1xf32>
        # CHECK:            %[[D31:.+]] = arith.addi {{.*}}, %[[C3]]
        # CHECK:            vector.store %[[D30]], {{.*}}[%[[D31]], {{.*}}] : memref<128x128xf32, strided<[128, 1], offset:
        # CHECK-SAME:         ?>>, vector<1xf32>
