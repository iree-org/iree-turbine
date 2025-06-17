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
    # CHECK-DAG:    #[[MAP0:.*]] = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #[[MAP2:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #[[MAP4:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK-DAG:    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [128, 2, 1] subgroup_size = 64>
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
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: 32,
        N: 32,
        K: 256,
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
    # CHECK-DAG:   #[[MAP:.*]] = affine_map<()[s0, s1] -> ((s1 * 16 + s0 floordiv 8) mod 32)>
    # CHECK-DAG:   #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 8) * 128)>
    # CHECK-DAG:   #[[MAP2:.*]] = affine_map<()[s0] -> (s0 mod 32)>
    # CHECK-DAG:   #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:   #[[MAP4:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:   #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
    # CHECK-DAG:   #[[MAP6:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:   #[[MAP7:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG:   #[[MAP8:.*]] = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:   #[[MAP9:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:   #[[MAP10:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:   #[[MAP11:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:   #[[MAP12:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-NEXT:   %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-NEXT:   %[[C9216:.*]] = arith.constant 9216 : index
    # CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C8704:.*]] = arith.constant 8704 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<9728xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C8704]]][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_0:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C9216]]][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C4352]]][] : memref<9728xi8, #gpu.address_space<workgroup>> to memref<32x136xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x128xi8, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY0:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD0:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x128xi8, strided<[128, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[LOAD0]], %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD1:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY2]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD1]], %[[VIEW_1]][%[[AFFINE_APPLY2]], %[[C0]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x128xi8, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD2:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x128xi8, strided<[128, 1], offset: ?>>, vector<16xi8>
    # CHECK-NEXT:   vector.store %[[LOAD2]], %[[VIEW_0]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD3:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY2]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD3]], %[[VIEW]][%[[AFFINE_APPLY2]], %[[C0]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD4:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY4]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY5:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD5:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY5]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY6:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD6:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY6]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[AFFINE_APPLY7:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD7:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY7]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[AFFINE_APPLY8:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD8:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY4]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD9:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY5]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD10:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY6]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[LOAD11:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY7]]] : memref<32x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-NEXT:   %[[BITCAST0:.*]] = vector.bitcast %[[LOAD10]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[BITCAST1:.*]] = vector.bitcast %[[LOAD11]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[BITCAST2:.*]] = vector.bitcast %[[LOAD6]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[BITCAST3:.*]] = vector.bitcast %[[LOAD7]] : vector<16xi8> to vector<32xf4E2M1FN>
    # CHECK-NEXT:   %[[EXTRACT0:.*]] = vector.extractelement %[[LOAD8]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT1:.*]] = vector.extractelement %[[LOAD4]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA0:.*]] = amdgpu.scaled_mfma(%[[EXTRACT0]][0] * %[[BITCAST0]]) * (%[[EXTRACT1]][0] * %[[BITCAST2]]) + %[[CST]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT2:.*]] = vector.extractelement %[[LOAD9]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT3:.*]] = vector.extractelement %[[LOAD5]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA1:.*]] = amdgpu.scaled_mfma(%[[EXTRACT2]][0] * %[[BITCAST1]]) * (%[[EXTRACT3]][0] * %[[BITCAST3]]) + %[[SCALED_MFMA0]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf4E2M1FN>, f8E8M0FNU, vector<32xf4E2M1FN>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE0:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY9:.*]] = affine.apply #[[MAP9]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE0]], %[[SPAN4]][%[[AFFINE_APPLY9]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE1:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY10:.*]] = affine.apply #[[MAP10]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE1]], %[[SPAN4]][%[[AFFINE_APPLY10]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE2:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY11:.*]] = affine.apply #[[MAP11]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE2]], %[[SPAN4]][%[[AFFINE_APPLY11]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE3:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY12:.*]] = affine.apply #[[MAP12]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE3]], %[[SPAN4]][%[[AFFINE_APPLY12]], %[[AFFINE_APPLY3]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
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
        BLOCK_K: 256,
        M: 32,
        N: 32,
        K: 256,
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
    # CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 32 + 128)>
    # CHECK-DAG: #[[MAP3:.*]] = affine_map<()[s0] -> (s0 mod 32)>
    # CHECK-DAG: #[[MAP4:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG: #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG: #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
    # CHECK-DAG: #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG: #[[MAP8:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG: #[[MAP9:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG: #[[MAP10:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-NEXT:   %[[C8448:.*]] = arith.constant 8448 : index
    # CHECK-NEXT:   %[[C17408:.*]] = arith.constant 17408 : index
    # CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C16896:.*]] = arith.constant 16896 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<17920xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C16896]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_0:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x264xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C17408]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C8448]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x264xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY0:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD0:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD1:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY2]]] : memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD0]], %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD1]], %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY2]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD2:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY3]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD2]], %[[VIEW_1]][%[[AFFINE_APPLY3]], %[[C0]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD3:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD4:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x256xf8E5M2, strided<[256, 1], offset: ?>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD3]], %[[VIEW_0]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD4]], %[[VIEW_0]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD5:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY3]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD5]], %[[VIEW]][%[[AFFINE_APPLY3]], %[[C0]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[AFFINE_APPLY5:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD6:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY5]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY6:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD7:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY6]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD8:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD9:.*]] = vector.load %[[VIEW_0]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD10:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY5]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD11:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY6]]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD12:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD13:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY2]]] : memref<32x264xf8E5M2, #gpu.address_space<workgroup>>, vector<32xf8E5M2>
    # CHECK-NEXT:   %[[EXTRACT0:.*]] = vector.extractelement %[[LOAD10]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT1:.*]] = vector.extractelement %[[LOAD6]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA0:.*]] = amdgpu.scaled_mfma(%[[EXTRACT0]][0] * %[[LOAD12]]) * (%[[EXTRACT1]][0] * %[[LOAD8]]) + %[[CST]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf8E5M2>, f8E8M0FNU, vector<32xf8E5M2>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT2:.*]] = vector.extractelement %[[LOAD11]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT3:.*]] = vector.extractelement %[[LOAD7]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA1:.*]] = amdgpu.scaled_mfma(%[[EXTRACT2]][0] * %[[LOAD13]]) * (%[[EXTRACT3]][0] * %[[LOAD9]]) + %[[SCALED_MFMA0]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf8E5M2>, f8E8M0FNU, vector<32xf8E5M2>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE0:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY7:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE0]], %[[SPAN4]][%[[AFFINE_APPLY7]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE1:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY8:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE1]], %[[SPAN4]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE2:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY9:.*]] = affine.apply #[[MAP9]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE2]], %[[SPAN4]][%[[AFFINE_APPLY9]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE3:.*]] = vector.extract_strided_slice %[[SCALED_MFMA1]] {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY10:.*]] = affine.apply #[[MAP10]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE3]], %[[SPAN4]][%[[AFFINE_APPLY10]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
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
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: 32,
        N: 32,
        K: 256,
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
    # CHECK-DAG:    #[[MAP:.*]] = affine_map<()[s0, s1] -> ((s1 * 8 + s0 floordiv 16) mod 32)>
    # CHECK-DAG:    #[[MAP1:.*]] = affine_map<()[s0] -> (s0 * 16 - (s0 floordiv 16) * 256)>
    # CHECK-DAG:    #[[MAP2:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + 16)>
    # CHECK-DAG:    #[[MAP3:.*]] = affine_map<()[s0, s1] -> (s1 * 8 + s0 floordiv 16 - ((s1 * 8 + s0 floordiv 16 + 16) floordiv 32) * 32 + 16)>
    # CHECK-DAG:    #[[MAP4:.*]] = affine_map<()[s0] -> (s0 mod 32)>
    # CHECK-DAG:    #[[MAP5:.*]] = affine_map<()[s0] -> (s0 * 16 + 16)>
    # CHECK-DAG:    #[[MAP6:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 32) * 32)>
    # CHECK-DAG:    #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32)>
    # CHECK-DAG:    #[[MAP8:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 2)>
    # CHECK-DAG:    #[[MAP9:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 4)>
    # CHECK-DAG:    #[[MAP10:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 32 + 6)>
    # CHECK-DAG:    #[[MAP11:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 32)>
    # CHECK-DAG:    #[[MAP12:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 32 + 64)>
    # CHECK-DAG:    #[[MAP13:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 32 + 128)>
    # CHECK-DAG:    #[[MAP14:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 32) * 32 + 192)>
    # CHECK-DAG:    #[[MAP15:.*]] = affine_map<()[s0] -> (s0 mod 32 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #[[MAP16:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4)>
    # CHECK-DAG:    #[[MAP17:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 1)>
    # CHECK-DAG:    #[[MAP18:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 2)>
    # CHECK-DAG:    #[[MAP19:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 3)>
    # CHECK-DAG:    #[[MAP20:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 8)>
    # CHECK-DAG:    #[[MAP21:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 9)>
    # CHECK-DAG:    #[[MAP22:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 10)>
    # CHECK-DAG:    #[[MAP23:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 32) * 4 + 11)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-NEXT:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[CST_0:.*]] = arith.constant dense<5.877470e-39> : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[CST_1:.*]] = arith.constant dense<0.000000e+00> : vector<16xf32>
    # CHECK-NEXT:   %[[C8448:.*]] = arith.constant 8448 : index
    # CHECK-NEXT:   %[[C17408:.*]] = arith.constant 17408 : index
    # CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-NEXT:   %[[C16896:.*]] = arith.constant 16896 : index
    # CHECK-NEXT:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<17920xi8, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C16896]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_3:.*]] = memref.view %[[ALLOC]][%[[C17408]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[VIEW_4:.*]] = memref.view %[[ALLOC]][%[[C8448]]][] : memref<17920xi8, #gpu.address_space<workgroup>> to memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>
    # CHECK-NEXT:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY0:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD0:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI0:.*]] = arith.cmpi slt, %[[AFFINE_APPLY0]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[SPLAT0:.*]] = vector.splat %[[CMPI0]] : vector<16xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW_4]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]], %[[SPLAT0]], %[[LOAD0]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<16xi1>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD1:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[CMPI1:.*]] = arith.cmpi slt, %[[AFFINE_APPLY3]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[SPLAT1:.*]] = vector.splat %[[CMPI1]] : vector<16xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW_4]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]], %[[SPLAT1]], %[[LOAD1]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<16xi1>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD2:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY4]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   %[[CMPI2:.*]] = arith.cmpi slt, %[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[SPLAT2:.*]] = vector.splat %[[CMPI2]] : vector<8xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW_3]][%[[AFFINE_APPLY4]], %[[C0]]], %[[SPLAT2]], %[[LOAD2]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xi1>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD3:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY5:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[CMPI3:.*]] = arith.cmpi slt, %[[AFFINE_APPLY0]], %[[AFFINE_APPLY5]] : index
    # CHECK-NEXT:   %[[SPLAT3:.*]] = vector.splat %[[CMPI3]] : vector<16xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW_2]][%[[AFFINE_APPLY0]], %[[AFFINE_APPLY1]]], %[[SPLAT3]], %[[LOAD3]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<16xi1>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[LOAD4:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]] : memref<32x256xf8E4M3FN, strided<[256, 1], offset: ?>>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[CMPI4:.*]] = arith.cmpi slt, %[[AFFINE_APPLY3]], %[[AFFINE_APPLY5]] : index
    # CHECK-NEXT:   %[[SPLAT4:.*]] = vector.splat %[[CMPI4]] : vector<16xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW_2]][%[[AFFINE_APPLY3]], %[[AFFINE_APPLY1]]], %[[SPLAT4]], %[[LOAD4]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<16xi1>, vector<16xf8E4M3FN>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD5:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY4]], %[[C0]]] : memref<32x8xf8E8M0FNU, strided<[8, 1], offset: ?>>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   %[[CMPI5:.*]] = arith.cmpi slt, %[[AFFINE_APPLY4]], %[[AFFINE_APPLY5]] : index
    # CHECK-NEXT:   %[[SPLAT5:.*]] = vector.splat %[[CMPI5]] : vector<8xi1>
    # CHECK-NEXT:   vector.maskedstore %[[VIEW]][%[[AFFINE_APPLY4]], %[[C0]]], %[[SPLAT5]], %[[LOAD5]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8xi1>, vector<8xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[AFFINE_APPLY6:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY7:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI6:.*]] = arith.cmpi slt, %[[AFFINE_APPLY6]], %[[AFFINE_APPLY5]] : index
    # CHECK-NEXT:   %[[SPLAT6:.*]] = vector.splat %[[CMPI6]] : vector<1xi1>
    # CHECK-NEXT:   %[[MASKED_LOAD0:.*]] = vector.maskedload %[[VIEW]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY7]]], %[[SPLAT6]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY8:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD1:.*]] = vector.maskedload %[[VIEW]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY8]]], %[[SPLAT6]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY9:.*]] = affine.apply #[[MAP9]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD2:.*]] = vector.maskedload %[[VIEW]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY9]]], %[[SPLAT6]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY10:.*]] = affine.apply #[[MAP10]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD3:.*]] = vector.maskedload %[[VIEW]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY10]]], %[[SPLAT6]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[AFFINE_APPLY11:.*]] = affine.apply #[[MAP11]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[SPLAT7:.*]] = vector.splat %[[CMPI6]] : vector<32xi1>
    # CHECK-NEXT:   %[[MASKED_LOAD4:.*]] = vector.maskedload %[[VIEW_2]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY11]]], %[[SPLAT7]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY12:.*]] = affine.apply #[[MAP12]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD5:.*]] = vector.maskedload %[[VIEW_2]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY12]]], %[[SPLAT7]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY13:.*]] = affine.apply #[[MAP13]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD6:.*]] = vector.maskedload %[[VIEW_2]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY13]]], %[[SPLAT7]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY14:.*]] = affine.apply #[[MAP14]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[MASKED_LOAD7:.*]] = vector.maskedload %[[VIEW_2]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY14]]], %[[SPLAT7]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[AFFINE_APPLY15:.*]] = affine.apply #[[MAP15]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI7:.*]] = arith.cmpi slt, %[[AFFINE_APPLY15]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[SPLAT8:.*]] = vector.splat %[[CMPI7]] : vector<1xi1>
    # CHECK-NEXT:   %[[MASKED_LOAD8:.*]] = vector.maskedload %[[VIEW_3]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY7]]], %[[SPLAT8]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[MASKED_LOAD9:.*]] = vector.maskedload %[[VIEW_3]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY8]]], %[[SPLAT8]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[MASKED_LOAD10:.*]] = vector.maskedload %[[VIEW_3]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY9]]], %[[SPLAT8]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[MASKED_LOAD11:.*]] = vector.maskedload %[[VIEW_3]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY10]]], %[[SPLAT8]], %[[CST_0]] : memref<32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xi1>, vector<1xf8E8M0FNU> into vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPLAT9:.*]] = vector.splat %[[CMPI7]] : vector<32xi1>
    # CHECK-NEXT:   %[[MASKED_LOAD12:.*]] = vector.maskedload %[[VIEW_4]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY11]]], %[[SPLAT9]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[MASKED_LOAD13:.*]] = vector.maskedload %[[VIEW_4]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY12]]], %[[SPLAT9]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[MASKED_LOAD14:.*]] = vector.maskedload %[[VIEW_4]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY13]]], %[[SPLAT9]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[MASKED_LOAD15:.*]] = vector.maskedload %[[VIEW_4]][%[[AFFINE_APPLY15]], %[[AFFINE_APPLY14]]], %[[SPLAT9]], %[[CST]] : memref<32x264xf8E4M3FN, #gpu.address_space<workgroup>>, vector<32xi1>, vector<32xf8E4M3FN> into vector<32xf8E4M3FN>
    # CHECK-NEXT:   %[[EXTRACT0:.*]] = vector.extractelement %[[MASKED_LOAD8]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT1:.*]] = vector.extractelement %[[MASKED_LOAD0]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA0:.*]] = amdgpu.scaled_mfma(%[[EXTRACT0]][0] * %[[MASKED_LOAD12]]) * (%[[EXTRACT1]][0] * %[[MASKED_LOAD4]]) + %[[CST_1]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf8E4M3FN>, f8E8M0FNU, vector<32xf8E4M3FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT2:.*]] = vector.extractelement %[[MASKED_LOAD9]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT3:.*]] = vector.extractelement %[[MASKED_LOAD1]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA1:.*]] = amdgpu.scaled_mfma(%[[EXTRACT2]][0] * %[[MASKED_LOAD13]]) * (%[[EXTRACT3]][0] * %[[MASKED_LOAD5]]) + %[[SCALED_MFMA0]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf8E4M3FN>, f8E8M0FNU, vector<32xf8E4M3FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT4:.*]] = vector.extractelement %[[MASKED_LOAD10]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT5:.*]] = vector.extractelement %[[MASKED_LOAD2]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA2:.*]] = amdgpu.scaled_mfma(%[[EXTRACT4]][0] * %[[MASKED_LOAD14]]) * (%[[EXTRACT5]][0] * %[[MASKED_LOAD6]]) + %[[SCALED_MFMA1]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf8E4M3FN>, f8E8M0FNU, vector<32xf8E4M3FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT6:.*]] = vector.extractelement %[[MASKED_LOAD11]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACT7:.*]] = vector.extractelement %[[MASKED_LOAD3]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA3:.*]] = amdgpu.scaled_mfma(%[[EXTRACT6]][0] * %[[MASKED_LOAD15]]) * (%[[EXTRACT7]][0] * %[[MASKED_LOAD7]]) + %[[SCALED_MFMA2]] {k = 64 : i32, m = 32 : i32, n = 32 : i32} : f8E8M0FNU, vector<32xf8E4M3FN>, f8E8M0FNU, vector<32xf8E4M3FN>, vector<16xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE0:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [0], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY16:.*]] = affine.apply #[[MAP16]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI8:.*]] = arith.cmpi slt, %[[AFFINE_APPLY16]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND0:.*]] = arith.andi %[[CMPI6]], %[[CMPI8]] : i1
    # CHECK-NEXT:   %[[SPLAT10:.*]] = vector.splat %[[AND0]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY16]], %[[AFFINE_APPLY6]]], %[[SPLAT10]], %[[EXTRACT_STRIDED_SLICE0]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE1:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [1], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY17:.*]] = affine.apply #[[MAP17]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI9:.*]] = arith.cmpi slt, %[[AFFINE_APPLY17]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND1:.*]] = arith.andi %[[CMPI6]], %[[CMPI9]] : i1
    # CHECK-NEXT:   %[[SPLAT11:.*]] = vector.splat %[[AND1]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY17]], %[[AFFINE_APPLY6]]], %[[SPLAT11]], %[[EXTRACT_STRIDED_SLICE1]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE2:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [2], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY18:.*]] = affine.apply #[[MAP18]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI10:.*]] = arith.cmpi slt, %[[AFFINE_APPLY18]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND2:.*]] = arith.andi %[[CMPI6]], %[[CMPI10]] : i1
    # CHECK-NEXT:   %[[SPLAT12:.*]] = vector.splat %[[AND2]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY18]], %[[AFFINE_APPLY6]]], %[[SPLAT12]], %[[EXTRACT_STRIDED_SLICE2]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE3:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [3], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY19:.*]] = affine.apply #[[MAP19]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI11:.*]] = arith.cmpi slt, %[[AFFINE_APPLY19]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND3:.*]] = arith.andi %[[CMPI6]], %[[CMPI11]] : i1
    # CHECK-NEXT:   %[[SPLAT13:.*]] = vector.splat %[[AND3]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY19]], %[[AFFINE_APPLY6]]], %[[SPLAT13]], %[[EXTRACT_STRIDED_SLICE3]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE4:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [4], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY20:.*]] = affine.apply #[[MAP20]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI12:.*]] = arith.cmpi slt, %[[AFFINE_APPLY20]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND4:.*]] = arith.andi %[[CMPI6]], %[[CMPI12]] : i1
    # CHECK-NEXT:   %[[SPLAT14:.*]] = vector.splat %[[AND4]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY20]], %[[AFFINE_APPLY6]]], %[[SPLAT14]], %[[EXTRACT_STRIDED_SLICE4]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE5:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [5], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY21:.*]] = affine.apply #[[MAP21]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI13:.*]] = arith.cmpi slt, %[[AFFINE_APPLY21]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND5:.*]] = arith.andi %[[CMPI6]], %[[CMPI13]] : i1
    # CHECK-NEXT:   %[[SPLAT15:.*]] = vector.splat %[[AND5]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY21]], %[[AFFINE_APPLY6]]], %[[SPLAT15]], %[[EXTRACT_STRIDED_SLICE5]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE6:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [6], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY22:.*]] = affine.apply #[[MAP22]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI14:.*]] = arith.cmpi slt, %[[AFFINE_APPLY22]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND6:.*]] = arith.andi %[[CMPI6]], %[[CMPI14]] : i1
    # CHECK-NEXT:   %[[SPLAT16:.*]] = vector.splat %[[AND6]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY22]], %[[AFFINE_APPLY6]]], %[[SPLAT16]], %[[EXTRACT_STRIDED_SLICE6]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE7:.*]] = vector.extract_strided_slice %[[SCALED_MFMA3]] {offsets = [7], sizes = [1], strides = [1]} : vector<16xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY23:.*]] = affine.apply #[[MAP23]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[CMPI15:.*]] = arith.cmpi slt, %[[AFFINE_APPLY23]], %[[AFFINE_APPLY2]] : index
    # CHECK-NEXT:   %[[AND7:.*]] = arith.andi %[[CMPI6]], %[[CMPI15]] : i1
    # CHECK-NEXT:   %[[SPLAT17:.*]] = vector.splat %[[AND7]] : vector<1xi1>
    # CHECK-NEXT:   vector.maskedstore %[[SPAN4]][%[[AFFINE_APPLY23]], %[[AFFINE_APPLY6]]], %[[SPLAT17]], %[[EXTRACT_STRIDED_SLICE7]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xi1>, vector<1xf32>
    # CHECK-NEXT:   return
