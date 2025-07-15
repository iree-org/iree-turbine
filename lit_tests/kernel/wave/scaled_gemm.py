# RUN: python %s | FileCheck %s

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType

from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)
from iree.turbine.kernel.wave.constraints import (
    ScaledMMAType,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


@run_test
def test_scaled_gemm_mxfp4():
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
    def scaled_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 1024,
        N: 1024,
        K: 1024,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        backend="rocm",
        target="gfx950",
    )

    scaled_gemm = wave_compile(options, scaled_gemm)
    print(scaled_gemm.asm)

    # CHECK-LABEL: test_scaled_gemm_mxfp4
    # CHECK-DAG:    #map = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map3 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map4 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map5 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map6 = affine_map<()[s0, s1] -> (s0 * 64 + ((s1 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map7 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map8 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #map9 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #map10 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #map11 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @scaled_gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              vector.load
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-4:      vector.load
    # CHECK:              amdgpu.scaled_mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def test_scaled_gemm_mxfp8():
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
    def scaled_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f8e5m2],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f8e5m2],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.f8e8m0fnu],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_scale_reg = tkw.read(a_scale)
            b_reg = tkw.read(b)
            b_scale_reg = tkw.read(b_scale)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 128,
        M: 1024,
        N: 1024,
        K: 1024,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        backend="rocm",
        target="gfx950",
    )

    scaled_gemm = wave_compile(options, scaled_gemm)
    print(scaled_gemm.asm)

    # CHECK-LABEL: test_scaled_gemm_mxfp8
    # CHECK-DAG:    #map = affine_map<()[s0, s1] -> (s0 + s1 * 32 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map1 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
    # CHECK-DAG:    #map2 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map3 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG:    #map4 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map5 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map6 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
    # CHECK-DAG:    #map7 = affine_map<()[s0, s1] -> (s0 * 128 + ((s1 mod 64) floordiv 16) * 16)>
    # CHECK-DAG:    #map8 = affine_map<()[s0, s1] -> (s0 * 128 + ((s1 mod 64) floordiv 16) * 16 + 64)>
    # CHECK-DAG:    #map9 = affine_map<()[s0, s1] -> (s1 * 4 + (s0 mod 64) floordiv 16)>
    # CHECK-DAG:    #map10 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:    #map11 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
    # CHECK-DAG:    #map12 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
    # CHECK-DAG:    #map13 = affine_map<()[s0, s1] -> (s0 * 32 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
    # CHECK:          func.func @scaled_gemm
    # CHECK-COUNT-1:    memref.alloc()
    # CHECK:            scf.for
    # CHECK:              vector.load
    # CHECK:              amdgpu.lds_barrier
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              vector.load
    # CHECK:              vector.store
    # CHECK:              amdgpu.lds_barrier
    # CHECK-COUNT-6:      vector.load
    # CHECK:              amdgpu.scaled_mfma
    # CHECK-COUNT-4:    vector.store


@run_test
def packed_mxfp4_test():
    mfma_variant = tkw.ScaledMMAType.F32_16x16x128_F8F6F4
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

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(4, 2, 1), mma_type=mfma_variant
        )
    ]

    @tkw.wave(constraints)
    def gemm_mxfp4_prefetch(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    shape = (1024, 1024, 1024)
    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
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
    }
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        compile_to_mlir=True,
    )
    gemm_mxfp4_prefetch = wave_compile(options, gemm_mxfp4_prefetch)
    print(gemm_mxfp4_prefetch.asm)

    # CHECK-LABEL:    gemm_mxfp4_prefetch

    # Prologue Global Read
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # CHECK:          vector.load {{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # CHECK-COUNT-4:  vector.load {{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # CHECK:          vector.load {{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>

    # Prologue Local Write
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK-COUNT-4:  vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:          vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>

    # Steady State
    # CHECK:          scf.for

    # Steady State global_load_rhs_scale
    # CHECK:            vector.load %{{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # Steady State local_load_rhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_lhs_scale
    # CHECK:            vector.load %{{.*}} : memref<1024x32xi8, strided<[32, 1], offset: ?>>, vector<4xi8>
    # Steady State local_load_lhs_scale
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>

    # Steady State global_load_rhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # Steady State local_load_rhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State global_load_lhs
    # CHECK-COUNT-4:    vector.load %{{.*}} : memref<1024x512xi8, strided<[512, 1], offset: ?>>, vector<16xi8>
    # Steady State local_load_lhs
    # CHECK=COUNT-16:   vector.load %{{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Steady State MFMA
    # CHECK-COUNT-64:   amdgpu.scaled_mfma

    # Steady State Local Write (lhs_scale, rhs_scale, lhs, rhs)
    # CHECK-COUNT-2:    vector.store {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<4xi8>
    # CHECK-COUNT-8:    vector.store {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK:            scf.yield
    # CHECK:          }

    # Epilogue Local Read
    # CHECK-COUNT-16: vector.load {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-16: vector.load {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<256x16xi8, #gpu.address_space<workgroup>>, vector<1xi8>
    # CHECK-COUNT-8:  vector.load {{.*}} : memref<256x136xi8, #gpu.address_space<workgroup>>, vector<16xi8>

    # Epilogue MFMA
    # CHECK-COUNT-64: amdgpu.scaled_mfma
