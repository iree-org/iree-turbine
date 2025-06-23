# RUN: python %s | FileCheck %s

from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
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

    compile_options = WaveCompileOptions(
        subs={
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
        compile_to_mlir=True,
    )
    mma = wave_compile(compile_options, mma)
    print(mma.asm)

    # CHECK-LABEL: test_mma
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #[[MAP2:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @mma
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-DAG:        %[[workgroup_id_0:.*]] = gpu.block_id x
    # CHECK-DAG:        %[[thread_id_x:.*]] = gpu.thread_id  x

    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<2560xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1280:.+]] = arith.constant 1280 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1280]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x20xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x20xf16,
    # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x20xf16,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D20]] * %[[D12]] + %[[CST]] {blocks = 1 : i32, k = 16 : i32, m = 16 :
    # CHECK-SAME:         i32, n = 16 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<4xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>

    # CHECK:            %[[OFF0:.*]] = affine.apply #[[MAP0]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D22]], {{.*}}[%[[OFF0]], {{.*}}]  : memref<64x128xf32, strided<[128, 1], offset:
    # CHECK-SAME:         ?>>, vector<1xf32>
    # CHECK:            %[[D26:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [1], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF1:.*]] = affine.apply #[[MAP1]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D26]], {{.*}}[%[[OFF1]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
    # CHECK-SAME:         ?>>, vector<1xf32>
    # CHECK:            %[[D28:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [2], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF2:.*]] = affine.apply #[[MAP2]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D28]], {{.*}}[%[[OFF2]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
    # CHECK-SAME:         ?>>, vector<1xf32>
    # CHECK:            %[[D30:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [3], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>
    # CHECK:            %[[OFF3:.*]] = affine.apply #[[MAP3]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[D30]], {{.*}}[%[[OFF3]], {{.*}}] : memref<64x128xf32, strided<[128, 1], offset:
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

    compile_options = WaveCompileOptions(
        subs={
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
        compile_to_mlir=True,
    )
    mma_32x32x8 = wave_compile(compile_options, mma_32x32x8)
    print(mma_32x32x8.asm)

    # CHECK-LABEL:    test_mma_32x32x8
    # CHECK-DAG:        #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:        #[[MAP1:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:        #[[MAP2:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:        #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:        #[[MAP4:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:        #[[MAP5:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:        #[[MAP6:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:        #[[MAP7:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:        #[[MAP8:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:        #[[MAP9:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:        #[[MAP10:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:        #[[MAP11:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:        #[[MAP12:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:        #[[MAP13:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:        #[[MAP14:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:        #[[MAP15:.*]] = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:          func.func @mma_32x32x8
    # CHECK-DAG:        %[[workgroup_id_0:.*]] = gpu.block_id x
    # CHECK-DAG:        %[[thread_id_x:.*]] = gpu.thread_id  x

    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1536:.+]] = arith.constant 1536 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x12xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1536]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x12xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x12xf16,
    # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x12xf16,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D20]] * %[[D12]] + %[[CST]] {blocks = 1 : i32, k = 8 : i32, m = 32 :
    # CHECK-SAME:         i32, n = 32 : i32} blgp =  none : vector<4xf16>, vector<4xf16>, vector<16xf32>
    # CHECK:            %[[VAL0:.*]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK:            %[[D23:.+]] = stream.binding.subspan {{.*}} : !stream.binding -> memref<128x128xf32,
    # CHECK-SAME:         strided<[128, 1], offset: ?>>

    # CHECK:            %[[OFF0:.*]] = affine.apply #[[MAP0]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL0]], %[[D23]][%[[OFF0]], %{{.*}}]
    # CHECK:            %[[VAL1:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [1], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF1:.*]] = affine.apply #[[MAP1]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL1]], %[[D23]][%[[OFF1]], %{{.*}}]
    # CHECK:            %[[VAL2:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [2], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF2:.*]] = affine.apply #[[MAP2]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL2]], %[[D23]][%[[OFF2]], %{{.*}}]
    # CHECK:            %[[VAL3:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [3], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF3:.*]] = affine.apply #[[MAP3]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL3]], %[[D23]][%[[OFF3]], %{{.*}}]
    # CHECK:            %[[VAL4:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [4], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF4:.*]] = affine.apply #[[MAP4]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL4]], %[[D23]][%[[OFF4]], %{{.*}}]
    # CHECK:            %[[VAL5:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [5], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF5:.*]] = affine.apply #[[MAP5]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL5]], %[[D23]][%[[OFF5]], %{{.*}}]
    # CHECK:            %[[VAL6:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [6], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF6:.*]] = affine.apply #[[MAP6]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL6]], %[[D23]][%[[OFF6]], %{{.*}}]
    # CHECK:            %[[VAL7:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [7], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF7:.*]] = affine.apply #[[MAP7]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL7]], %[[D23]][%[[OFF7]], %{{.*}}]
    # CHECK:            %[[VAL8:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [8], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF8:.*]] = affine.apply #[[MAP8]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL8]], %[[D23]][%[[OFF8]], %{{.*}}]
    # CHECK:            %[[VAL9:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [9], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF9:.*]] = affine.apply #[[MAP9]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL9]], %[[D23]][%[[OFF9]], %{{.*}}]
    # CHECK:            %[[VAL10:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [10], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF10:.*]] = affine.apply #[[MAP10]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL10]], %[[D23]][%[[OFF10]], %{{.*}}]
    # CHECK:            %[[VAL11:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [11], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF11:.*]] = affine.apply #[[MAP11]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL11]], %[[D23]][%[[OFF11]], %{{.*}}]
    # CHECK:            %[[VAL12:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [12], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF12:.*]] = affine.apply #[[MAP12]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL12]], %[[D23]][%[[OFF12]], %{{.*}}]
    # CHECK:            %[[VAL13:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [13], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF13:.*]] = affine.apply #[[MAP13]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL13]], %[[D23]][%[[OFF13]], %{{.*}}]
    # CHECK:            %[[VAL14:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [14], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF14:.*]] = affine.apply #[[MAP14]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL14]], %[[D23]][%[[OFF14]], %{{.*}}]
    # CHECK:            %[[VAL15:.*]] = vector.extract_strided_slice %[[D21]]  {offsets = [15], sizes = [1], strides = [1]}
    # CHECK:            %[[OFF15:.*]] = affine.apply #[[MAP15]]()[%[[workgroup_id_0]], %[[thread_id_x]]]
    # CHECK:            vector.store %[[VAL15]], %[[D23]][%[[OFF15]], %{{.*}}]


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

    compile_options = WaveCompileOptions(
        subs={
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
        compile_to_mlir=True,
    )
    mma_32x32x16 = wave_compile(compile_options, mma_32x32x16)
    print(mma_32x32x16.asm)

    # CHECK-LABEL:    test_mma_32x32x16
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 32 + ((s1 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:          func.func @mma_32x32x16

    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<3072xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1536:.+]] = arith.constant 1536 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x24xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1536]]][] : memref<3072xi8, #gpu.address_space<workgroup>> to memref<64x24xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]]{{.*}} : memref<64x24xf8E4M3FNUZ,
    # CHECK:            %[[D20:.+]] = vector.load %[[ALLOC_0]]{{.*}} : memref<64x24xf8E4M3FNUZ,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D20]] * %[[D12]] + %[[CST]] {blocks = 1 : i32, k = 16 : i32, m = 32 :
    # CHECK-SAME:         i32, n = 32 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<16xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<16xf32> to vector<1xf32>
    # CHECK:            %[[D23:.+]] = stream.binding.subspan {{.*}} : !stream.binding -> memref<128x128xf32,
    # CHECK-SAME:         strided<[128, 1], offset: ?>>

    # CHECK-COUNT-16:   vector.store


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

    compile_options = WaveCompileOptions(
        subs={
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
        compile_to_mlir=True,
    )
    mma_16x16x32 = wave_compile(compile_options, mma_16x16x32)
    print(mma_16x16x32.asm)

    # CHECK-LABEL:    test_mma_16x16x32
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:        #{{.*}} = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @mma_16x16x32
    # CHECK-DAG:        %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>

    # CHECK-DAG:        %[[BASE_ALLOC:.+]] = memref.alloc() : memref<2560xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[C0:.+]] = arith.constant 0 : index
    # CHECK-DAG:        %[[C1280:.+]] = arith.constant 1280 : index
    # CHECK-DAG:        %[[ALLOC:.+]] = memref.view %[[BASE_ALLOC]][%[[C0]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x40xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[ALLOC_0:.+]] = memref.view %[[BASE_ALLOC]][%[[C1280]]][] : memref<2560xi8, #gpu.address_space<workgroup>> to memref<32x40xf8E4M3FNUZ, #gpu.address_space<workgroup>>
    # CHECK-DAG:        %[[D12:.+]] = vector.load %[[ALLOC]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
    # CHECK-DAG:        %[[D20:.+]] = vector.load %[[ALLOC_0]][{{.*}}, {{.*}}] : memref<32x40xf8E4M3FNUZ,
    # CHECK:            %[[D21:.+]] = amdgpu.mfma %[[D20]] * %[[D12]] + %[[CST]] {blocks = 1 : i32, k = 32 : i32, m = 16 :
    # CHECK-SAME:         i32, n = 16 : i32} blgp =  none : vector<8xf8E4M3FNUZ>, vector<8xf8E4M3FNUZ>, vector<4xf32>
    # CHECK:            %[[D22:.+]] = vector.extract_strided_slice %[[D21]] {offsets = [0], sizes = [1], strides = [1]} :
    # CHECK-SAME:         vector<4xf32> to vector<1xf32>

    # CHECK-COUNT-4:    vector.store
