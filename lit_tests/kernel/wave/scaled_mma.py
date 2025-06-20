# RUN: python %s | FileCheck %s

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import ScaledMMAType
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


@run_test
def test_mxfp4_scaled_mma_16x16x128():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def scaled_mma(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a)
        a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
        a_scale_reg = tkw.read(a_scale)
        b_reg = tkw.read(b)
        b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
        b_scale_reg = tkw.read(b_scale)
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, c_reg)
        tkw.write(acc, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 32,
        N: 32,
        K: 128,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="rocm",
        target="gfx950",
    )
    scaled_mma = wave_compile(options, scaled_mma)
    print(scaled_mma.asm)

    # CHECK-LABEL:  test_mxfp4_scaled_mma_16x16x128
    # CHECK-DAG:    #[[MAP0:.+]] = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #[[MAP1:.+]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #[[MAP2:.+]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #[[MAP3:.+]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #[[MAP4:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #[[MAP5:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #[[MAP6:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #[[MAP7:.+]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:    func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-NEXT:   %[[C2304:.+]] = arith.constant 2304 : index
    # CHECK-NEXT:   %[[C4992:.+]] = arith.constant 4992 : index
    # CHECK-NEXT:   %[[C0:.+]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C4608:.+]] = arith.constant 4608 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.+]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.+]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.+]] = memref.alloc() : memref<5376xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.+]] = memref.view %[[ALLOC]][%[[C4608]]][] : memref<5376xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_0:.+]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<5376xi8, #gpu.address_space<workgroup>> to memref<32x72xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_1:.+]] = memref.view %[[ALLOC]][%[[C4992]]][] : memref<5376xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.+]] = memref.view %[[ALLOC]][%[[C2304]]][] : memref<5376xi8, #gpu.address_space<workgroup>> to memref<32x72xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SUBSPAN_0:.+]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x64xi8, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_0:.+]] = affine.apply #[[MAP0]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY_1:.+]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[VECTOR_LOAD_0:.+]] = vector.load %[[SUBSPAN_0]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_1]]] : memref<32x64xi8, strided<[64, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[VECTOR_LOAD_0]], %[[VIEW_2]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_1]]] : memref<32x72xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SUBSPAN_1:.+]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_2:.+]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[VECTOR_LOAD_1:.+]] = vector.load %[[SUBSPAN_1]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_2]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[VECTOR_LOAD_1]], %[[VIEW_1]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SUBSPAN_2:.+]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x64xi8, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_3:.+]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[VECTOR_LOAD_2:.+]] = vector.load %[[SUBSPAN_2]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<32x64xi8, strided<[64, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[VECTOR_LOAD_2]], %[[VIEW_0]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<32x72xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SUBSPAN_3:.+]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[VECTOR_LOAD_3:.+]] = vector.load %[[SUBSPAN_3]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_2]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[VECTOR_LOAD_3]], %[[VIEW]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[VECTOR_LOAD_4:.+]] = vector.load %[[VIEW]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[VECTOR_LOAD_5:.+]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<32x72xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[VECTOR_LOAD_6:.+]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[VECTOR_LOAD_7:.+]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY_0]], %[[AFFINE_APPLY_1]]] : memref<32x72xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[BITCAST_0:.+]] = vector.bitcast %[[VECTOR_LOAD_7]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[BITCAST_1:.+]] = vector.bitcast %[[VECTOR_LOAD_5]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[EXTRACT_ELEMENT_0:.+]] = vector.extractelement %[[VECTOR_LOAD_6]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT_ELEMENT_1:.+]] = vector.extractelement %[[VECTOR_LOAD_4]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA:.+]] = amdgpu.scaled_mfma(%[[EXTRACT_ELEMENT_0]][0] * %[[BITCAST_0]]) * (%[[EXTRACT_ELEMENT_1]][0] * %[[BITCAST_1]]) + %[[CST]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_0:.+]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SUBSPAN_4:.+]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_4:.+]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_0]], %[[SUBSPAN_4]][%[[AFFINE_APPLY_4]], %[[AFFINE_APPLY_3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_1:.+]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_5:.+]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_1]], %[[SUBSPAN_4]][%[[AFFINE_APPLY_5]], %[[AFFINE_APPLY_3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_2:.+]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_6:.+]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_2]], %[[SUBSPAN_4]][%[[AFFINE_APPLY_6]], %[[AFFINE_APPLY_3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_3:.+]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_7:.+]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_3]], %[[SUBSPAN_4]][%[[AFFINE_APPLY_7]], %[[AFFINE_APPLY_3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   return


@run_test
def test_mxfp4_scaled_mma_32x32x64():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_32x32x64_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def scaled_mma(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a)
        a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
        a_scale_reg = tkw.read(a_scale)
        b_reg = tkw.read(b)
        b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
        b_scale_reg = tkw.read(b_scale)
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, c_reg)
        tkw.write(acc, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 64,
        M: 64,
        N: 64,
        K: 64,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="rocm",
        target="gfx950",
    )
    scaled_mma = wave_compile(options, scaled_mma)
    print(scaled_mma.asm)

    # CHECK-LABEL:  test_mxfp4_scaled_mma_32x32x64
    # CHECK-DAG:   #[[MAP:.*]] = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
    # CHECK-DAG:   #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 16)>
    # CHECK-DAG:   #[[MAP2:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
    # CHECK-DAG:   #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
    # CHECK-DAG:   #[[MAP4:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:   #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:   #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:   #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:   #[[MAP8:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:   #[[MAP9:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:   #[[MAP10:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:   #[[MAP11:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:   #[[MAP12:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:   #[[MAP13:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:   #[[MAP14:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:   #[[MAP15:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:   #[[MAP16:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:   #[[MAP17:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:   #[[MAP18:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:   #[[MAP19:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:    func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-NEXT:   %[[C2560:.*]] = arith.constant 2560 : index
    # CHECK-NEXT:   %[[C5760:.*]] = arith.constant 5760 : index
    # CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C5120:.*]] = arith.constant 5120 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.*]] = gpu.thread_id x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<6400xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C5120]]][] : memref<6400xi8, #gpu.address_space<workgroup>> to memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_0:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<6400xi8, #gpu.address_space<workgroup>> to memref<64x40xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C5760]]][] : memref<6400xi8, #gpu.address_space<workgroup>> to memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C2560]]][] : memref<6400xi8, #gpu.address_space<workgroup>> to memref<64x40xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<64x32xi8, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY_0:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_0]]] : memref<64x32xi8, strided<[32, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[LOAD]], %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_0]]] : memref<64x40xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_1:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD_2:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_1]]] : memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD_2]], %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_1]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<64x32xi8, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD_4:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_0]]] : memref<64x32xi8, strided<[32, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[LOAD_4]], %[[VIEW_0]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_0]]] : memref<64x40xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD_5:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD_5]], %[[VIEW]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[LOAD_6:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_1]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD_7:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY_3]], %[[AFFINE_APPLY_0]]] : memref<64x40xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[LOAD_8:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_1]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD_9:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_0]]] : memref<64x40xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[BITCAST:.*]] = vector.bitcast %[[LOAD_9]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[BITCAST_10:.*]] = vector.bitcast %[[LOAD_7]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[EXTRACTELEMENT:.*]] = vector.extractelement %[[LOAD_8]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACTELEMENT_11:.*]] = vector.extractelement %[[LOAD_6]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA:.*]] = amdgpu.scaled_mfma(%[[EXTRACTELEMENT]][0] * %[[BITCAST]]) * (%[[EXTRACTELEMENT_11]][0] * %[[BITCAST_10]]) + %[[CST]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<64x64xf32, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_12:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE]], %[[SPAN4]][%[[AFFINE_APPLY_12]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_13:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_14:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_13]], %[[SPAN4]][%[[AFFINE_APPLY_14]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_15:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_16:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_15]], %[[SPAN4]][%[[AFFINE_APPLY_16]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_17:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_18:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_17]], %[[SPAN4]][%[[AFFINE_APPLY_18]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_19:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_20:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_19]], %[[SPAN4]][%[[AFFINE_APPLY_20]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_21:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_22:.*]] = affine.apply #[[MAP9]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_21]], %[[SPAN4]][%[[AFFINE_APPLY_22]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_23:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_24:.*]] = affine.apply #[[MAP10]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_23]], %[[SPAN4]][%[[AFFINE_APPLY_24]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_25:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_26:.*]] = affine.apply #[[MAP11]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_25]], %[[SPAN4]][%[[AFFINE_APPLY_26]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_27:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_28:.*]] = affine.apply #[[MAP12]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_27]], %[[SPAN4]][%[[AFFINE_APPLY_28]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_29:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_30:.*]] = affine.apply #[[MAP13]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_29]], %[[SPAN4]][%[[AFFINE_APPLY_30]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_31:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_32:.*]] = affine.apply #[[MAP14]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_31]], %[[SPAN4]][%[[AFFINE_APPLY_32]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_33:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_34:.*]] = affine.apply #[[MAP15]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_33]], %[[SPAN4]][%[[AFFINE_APPLY_34]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_35:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_36:.*]] = affine.apply #[[MAP16]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_35]], %[[SPAN4]][%[[AFFINE_APPLY_36]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_37:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_38:.*]] = affine.apply #[[MAP17]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_37]], %[[SPAN4]][%[[AFFINE_APPLY_38]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_39:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_40:.*]] = affine.apply #[[MAP18]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_39]], %[[SPAN4]][%[[AFFINE_APPLY_40]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_41:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_42:.*]] = affine.apply #[[MAP19]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_41]], %[[SPAN4]][%[[AFFINE_APPLY_42]], %[[AFFINE_APPLY_3]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   return


@run_test
def test_mxfp8_scaled_mma_16x16x128():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def scaled_mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e5m2],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e5m2],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a)
        a_scale_reg = tkw.read(a_scale)
        b_reg = tkw.read(b)
        b_scale_reg = tkw.read(b_scale)
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, c_reg)
        tkw.write(acc, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 32,
        N: 32,
        K: 128,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="rocm",
        target="gfx950",
    )
    scaled_mma = wave_compile(options, scaled_mma)
    print(scaled_mma.asm)

    # CHECK-LABEL:  test_mxfp8_scaled_mma_16x16x128
    # CHECK-DAG: #[[MAP:.*]] = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 32)>
    # CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG: #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG: #[[MAP4:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG: #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG: #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG: #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-DAG:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-DAG:   %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-DAG:   %[[C9088:.*]] = arith.constant 9088 : index
    # CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:   %[[C8704:.*]] = arith.constant 8704 : index
    # CHECK:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<9472xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C8704]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_0:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x136xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C9088]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C4352]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x136xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD]], %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD1:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD1]], %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD2:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD2]], %[[VIEW_0]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD3:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY2]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD3]], %[[VIEW]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[LOAD4:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD5:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD6:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD7:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[EXTRACTELEMENT:.*]] = vector.extractelement %[[LOAD6]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACTELEMENT_3:.*]] = vector.extractelement %[[LOAD4]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA:.*]] = amdgpu.scaled_mfma(%[[EXTRACTELEMENT]][0] * %[[LOAD7]]) * (%[[EXTRACTELEMENT_3]][0] * %[[LOAD5]]) + %[[CST]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf8E5M2>, f8E8M0FNU, vector<32xf8E5M2>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE]], %[[SPAN4]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_4:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY5:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_4]], %[[SPAN4]][%[[AFFINE_APPLY5]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_5:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY6:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_5]], %[[SPAN4]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_6:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY7:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_6]], %[[SPAN4]][%[[AFFINE_APPLY7]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   return


@run_test
def test_mxfp8_scaled_mma_32x32x64():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    mfma_variant = ScaledMMAType.F32_32x32x64_F8F6F4

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(2, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def scaled_mma(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e4m3fn],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e4m3fn],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a)
        a_scale_reg = tkw.read(a_scale)
        b_reg = tkw.read(b)
        b_scale_reg = tkw.read(b_scale)
        c_reg = tkl.Register[M, N, tkl.f32](0.0)
        acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, c_reg)
        tkw.write(acc, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 64,
        M: 64,
        N: 64,
        K: 64,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        backend="rocm",
        target="gfx950",
    )
    scaled_mma = wave_compile(options, scaled_mma)
    print(scaled_mma.asm)

    # CHECK-LABEL:  test_mxfp8_scaled_mma_32x32x64
    # CHECK-DAG:    #[[MAP:.*]] = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 32)>
    # CHECK-DAG:    #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 32)>
    # CHECK-DAG:    #[[MAP2:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
    # CHECK-DAG:    #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 32) * 32)>
    # CHECK-DAG:    #[[MAP4:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:    #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:    #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:    #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:    #[[MAP8:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:    #[[MAP9:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:    #[[MAP10:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:    #[[MAP11:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    # CHECK-DAG:    #[[MAP12:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 16)>
    # CHECK-DAG:    #[[MAP13:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 17)>
    # CHECK-DAG:    #[[MAP14:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 18)>
    # CHECK-DAG:    #[[MAP15:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 19)>
    # CHECK-DAG:    #[[MAP16:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 24)>
    # CHECK-DAG:    #[[MAP17:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 25)>
    # CHECK-DAG:    #[[MAP18:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 26)>
    # CHECK-DAG:    #[[MAP19:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 32 + ((s0 mod 64) floordiv 32) * 4 + 27)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-NEXT:   %[[C4608:.*]] = arith.constant 4608 : index
    # CHECK-NEXT:   %[[C9856:.*]] = arith.constant 9856 : index
    # CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C9216:.*]] = arith.constant 9216 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<10496xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C9216]]][] : memref<10496xi8, #gpu.address_space<workgroup>> to memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_0:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<10496xi8, #gpu.address_space<workgroup>> to memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C9856]]][] : memref<10496xi8, #gpu.address_space<workgroup>> to memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C4608]]][] : memref<10496xi8, #gpu.address_space<workgroup>> to memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SPAN:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY_3:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD:.*]] = vector.load %[[SPAN]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_3]]] : memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   vector.store %[[LOAD]], %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_3]]] : memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[SPAN_4:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_5:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD_6:.*]] = vector.load %[[SPAN_4]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_5]]] : memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD_6]], %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_5]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN_7:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_8:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD_9:.*]] = vector.load %[[SPAN_7]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_3]]] : memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   vector.store %[[LOAD_9]], %[[VIEW_0]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_3]]] : memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[SPAN_10:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD_11:.*]] = vector.load %[[SPAN_10]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_5]]] : memref<64x2xf8E8M0FNU, strided<[2, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD_11]], %[[VIEW]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_5]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[LOAD_12:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_5]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD_13:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY_8]], %[[AFFINE_APPLY_3]]] : memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[LOAD_14:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_5]]] : memref<64x10xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD_15:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY_3]]] : memref<64x72xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[EXTRACTELEMEN:.*]] = vector.extractelement %[[LOAD_14]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACTELEMEN_16:.*]] = vector.extractelement %[[LOAD_12]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA:.*]] = amdgpu.scaled_mfma(%[[EXTRACTELEMEN]][0] * %[[LOAD_15]]) * (%[[EXTRACTELEMEN_16]][0] * %[[LOAD_13]]) + %[[CST]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf8E4M3FN>, f8E8M0FNU, vector<32xf8E4M3FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN_17:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<64x64xf32, strided<[64, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY_18:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE]], %[[SPAN_17]][%[[AFFINE_APPLY_18]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_19:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_20:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_19]], %[[SPAN_17]][%[[AFFINE_APPLY_20]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_21:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_22:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_21]], %[[SPAN_17]][%[[AFFINE_APPLY_22]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_23:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_24:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_23]], %[[SPAN_17]][%[[AFFINE_APPLY_24]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_25:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_26:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_25]], %[[SPAN_17]][%[[AFFINE_APPLY_26]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_27:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_28:.*]] = affine.apply #[[MAP9]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_27]], %[[SPAN_17]][%[[AFFINE_APPLY_28]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_29:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_30:.*]] = affine.apply #[[MAP10]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_29]], %[[SPAN_17]][%[[AFFINE_APPLY_30]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_31:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_32:.*]] = affine.apply #[[MAP11]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_31]], %[[SPAN_17]][%[[AFFINE_APPLY_32]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_33:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [8], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_34:.*]] = affine.apply #[[MAP12]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_33]], %[[SPAN_17]][%[[AFFINE_APPLY_34]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_35:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [9], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_36:.*]] = affine.apply #[[MAP13]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_35]], %[[SPAN_17]][%[[AFFINE_APPLY_36]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_37:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [10], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_38:.*]] = affine.apply #[[MAP14]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_37]], %[[SPAN_17]][%[[AFFINE_APPLY_38]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_39:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [11], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_40:.*]] = affine.apply #[[MAP15]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_39]], %[[SPAN_17]][%[[AFFINE_APPLY_40]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_41:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [12], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_42:.*]] = affine.apply #[[MAP16]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_41]], %[[SPAN_17]][%[[AFFINE_APPLY_42]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_43:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [13], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_44:.*]] = affine.apply #[[MAP17]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_43]], %[[SPAN_17]][%[[AFFINE_APPLY_44]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_45:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [14], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_46:.*]] = affine.apply #[[MAP18]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_45]], %[[SPAN_17]][%[[AFFINE_APPLY_46]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_47:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [15], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY_48:.*]] = affine.apply #[[MAP19]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_47]], %[[SPAN_17]][%[[AFFINE_APPLY_48]], %[[AFFINE_APPLY_8]]] : memref<64x64xf32, strided<[64, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   return
    # CHECK-NEXT: }
