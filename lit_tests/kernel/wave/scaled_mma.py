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

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

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
    # CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG: #[[MAP3:.*]] = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG: #[[MAP4:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG: #[[MAP5:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG: #[[MAP6:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG: #[[MAP7:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG: #[[MAP8:.*]] = affine_map<()[s0] -> ((s0 floordiv 64) * 16 + ((s0 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:   func.func @scaled_mma(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding, %arg3: !stream.binding, %arg4: !stream.binding) attributes {translation_info = #translation} {
    # CHECK-DAG:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK-DAG:   %[[CST_0:.*]] = arith.constant dense<0.000000e+00> : vector<32xf8E5M2>
    # CHECK-DAG:   %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-DAG:   %[[C9088:.*]] = arith.constant 9088 : index
    # CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:   %[[C8704:.*]] = arith.constant 8704 : index
    # CHECK:   %[[THREAD_ID_X:.*]] = gpu.thread_id  x
    # CHECK-NEXT:   %[[THREAD_ID_Y:.*]] = gpu.thread_id  y
    # CHECK-NEXT:   %[[ALLOC:.*]] = memref.alloc() : memref<9472xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW:.*]] = memref.view %[[ALLOC]][%[[C8704]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_1:.*]] = memref.view %[[ALLOC]][%[[C0]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x136xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_2:.*]] = memref.view %[[ALLOC]][%[[C9088]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>
    # CHECK-DAG:   %[[VIEW_3:.*]] = memref.view %[[ALLOC]][%[[C4352]]][] : memref<9472xi8, #gpu.address_space<workgroup>> to memref<32x136xf8E5M2, #gpu.address_space<workgroup>>
    # CHECK:   %[[SPAN0:.*]] = stream.binding.subspan %arg0[%[[C0]]] : !stream.binding -> memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY:.*]] = affine.apply #[[MAP]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[AFFINE_APPLY1:.*]] = affine.apply #[[MAP1]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[AFFINE_APPLY2:.*]] = affine.apply #[[MAP2]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD_0:.*]] = vector.load %[[SPAN0]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<16xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD]], %[[VIEW_3]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD_0]], %[[VIEW_3]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[SPAN1:.*]] = stream.binding.subspan %arg1[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY3:.*]] = affine.apply #[[MAP3]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   %[[LOAD1:.*]] = vector.load %[[SPAN1]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY3]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD1]], %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY3]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SPAN2:.*]] = stream.binding.subspan %arg2[%[[C0]]] : !stream.binding -> memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY4:.*]] = affine.apply #[[MAP4]]()[%[[THREAD_ID_X]], %[[THREAD_ID_Y]]]
    # CHECK-NEXT:   %[[LOAD2:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[LOAD_2:.*]] = vector.load %[[SPAN2]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x128xf8E5M2, strided<[128, 1], offset: ?>>, vector<16xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD2]], %[[VIEW_1]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   vector.store %[[LOAD_2]], %[[VIEW_1]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[SPAN3:.*]] = stream.binding.subspan %arg3[%[[C0]]] : !stream.binding -> memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>
    # CHECK-NEXT:   %[[LOAD3:.*]] = vector.load %[[SPAN3]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY3]]] : memref<32x4xf8E8M0FNU, strided<[4, 1], offset: ?>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   vector.store %[[LOAD3]], %[[VIEW]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY3]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   amdgpu.lds_barrier
    # CHECK-NEXT:   %[[LOAD4:.*]] = vector.load %[[VIEW]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY3]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD5:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[LOAD6:.*]] = vector.load %[[VIEW_1]][%[[AFFINE_APPLY4]], %[[AFFINE_APPLY2]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[INSERT_STRIDED_SLICE:.*]] = vector.insert_strided_slice %[[LOAD5]], %[[CST_0]] {offsets = [0], strides = [1]} : vector<16xf8E5M2> into vector<32xf8E5M2>
    # CHECK-NEXT:   %[[INSERT_STRIDED_SLICE_3:.*]] = vector.insert_strided_slice %[[LOAD6]], %[[INSERT_STRIDED_SLICE]] {offsets = [16], strides = [1]} : vector<16xf8E5M2> into vector<32xf8E5M2>
    # CHECK-NEXT:   %[[LOAD7:.*]] = vector.load %[[VIEW_2]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY3]]] : memref<32x12xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[LOAD8:.*]] = vector.load %[[VIEW_3]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY1]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[LOAD9:.*]] = vector.load %[[VIEW_3]][%[[AFFINE_APPLY]], %[[AFFINE_APPLY2]]] : memref<32x136xf8E5M2, #gpu.address_space<workgroup>>, vector<16xf8E5M2>
    # CHECK-NEXT:   %[[INSERT_STRIDED_SLICE_4:.*]] = vector.insert_strided_slice %[[LOAD8]], %[[CST_0]] {offsets = [0], strides = [1]} : vector<16xf8E5M2> into vector<32xf8E5M2>
    # CHECK-NEXT:   %[[INSERT_STRIDED_SLICE_5:.*]] = vector.insert_strided_slice %[[LOAD9]], %[[INSERT_STRIDED_SLICE_4]] {offsets = [16], strides = [1]} : vector<16xf8E5M2> into vector<32xf8E5M2>
    # CHECK-NEXT:   %[[EXTRACTELEMENT:.*]] = vector.extractelement %[[LOAD7]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[EXTRACTELEMENT_6:.*]] = vector.extractelement %[[LOAD4]][%[[C0]] : index] : vector<1xf8E8M0FNU>
    # CHECK-NEXT:   %[[SCALED_MFMA:.*]] = amdgpu.scaled_mfma(%[[EXTRACTELEMENT]][0] * %[[INSERT_STRIDED_SLICE_5]]) * (%[[EXTRACTELEMENT_6]][0] * %[[INSERT_STRIDED_SLICE_3]]) + %[[CST]] {k = 128 : i32, m = 16 : i32, n = 16 : i32} : f8E8M0FNU, vector<32xf8E5M2>, f8E8M0FNU, vector<32xf8E5M2>, vector<4xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[SPAN4:.*]] = stream.binding.subspan %arg4[%[[C0]]] : !stream.binding -> memref<32x32xf32, strided<[32, 1], offset: ?>>
    # CHECK-NEXT:   %[[AFFINE_APPLY5:.*]] = affine.apply #[[MAP5]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE]], %[[SPAN4]][%[[AFFINE_APPLY5]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_7:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY6:.*]] = affine.apply #[[MAP6]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_7]], %[[SPAN4]][%[[AFFINE_APPLY6]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_8:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY7:.*]] = affine.apply #[[MAP7]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_8]], %[[SPAN4]][%[[AFFINE_APPLY7]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   %[[EXTRACT_STRIDED_SLICE_9:.*]] = vector.extract_strided_slice %[[SCALED_MFMA]] {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
    # CHECK-NEXT:   %[[AFFINE_APPLY8:.*]] = affine.apply #[[MAP8]]()[%[[THREAD_ID_X]]]
    # CHECK-NEXT:   vector.store %[[EXTRACT_STRIDED_SLICE_9]], %[[SPAN4]][%[[AFFINE_APPLY8]], %[[AFFINE_APPLY4]]] : memref<32x32xf32, strided<[32, 1], offset: ?>>, vector<1xf32>
    # CHECK-NEXT:   return
