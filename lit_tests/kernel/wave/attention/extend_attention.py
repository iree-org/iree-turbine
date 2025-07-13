# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
import torch


@run_test
def test_extend_attention():
    shape = AttentionShape(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        num_seqs=2,
        max_seq_len=32,
        block_size=64,
    )
    total_token_num = 12189
    extend_token_num = 3198
    q_shape = (extend_token_num, shape.num_query_heads, shape.head_size)
    k_shape = (extend_token_num, shape.num_kv_heads, shape.head_size)
    v_shape = (extend_token_num, shape.num_kv_heads, shape.head_size_kv)
    o_shape = (extend_token_num, shape.num_query_heads, shape.head_size_kv)
    k_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    v_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
    )
    extend_attention = wave_compile(options, extend_attention)
    print(extend_attention.asm)

    # This part ensure correctness of WG distribution for extend attention.
    # CHECK-LABEL: test_extend_attention
    # CHECK-DAG:            #[[map0:.*]] = affine_map<()[s0] -> (s0 ceildiv 64)>
    # CHECK-DAG:            #[[map1:.*]] = affine_map<()[s0] -> (s0 * 16 - 16)>
    # CHECK:              stream.executable.export public @extend_attention workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
    # CHECK-DAG:            %[[C1:.*]] = arith.constant 1 : index
    # CHECK:                %[[NQ_GRID:.+]] = affine.apply #[[map0]]()[%[[ARG3]]]
    # CHECK:                %[[NUM_SEQ:.+]] = affine.apply #[[map1]]()[%[[ARG2]]]
    # CHECK:                stream.return %[[NQ_GRID]], %[[C1]], %[[NUM_SEQ]] : index, index, index

    # CHECK-LABEL:        func.func @extend_attention
    # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
    # CHECK-DAG:            %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[ALLOC0:.*]] = memref.alloc() : memref<8704xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC1:.*]] = memref.view %[[ALLOC0]][%[[C0]]][] : memref<8704xi8, #gpu.address_space<workgroup>> to memref<32x1x68xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC2:.*]] = memref.view %[[ALLOC0]][%[[C4352]]][] : memref<8704xi8, #gpu.address_space<workgroup>> to memref<1x32x68xf16, #gpu.address_space<workgroup>>
    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                scf.for
    # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
    # CHECK-COUNT-3:            vector.maskedload
    # CHECK-COUNT-2:            vector.maskedload
    # CHECK-NEXT:               vector.store %{{.*}}, %[[ALLOC2]]
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
    # CHECK-COUNT-32:           memref.load %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-2:            arith.cmpi slt
    # CHECK-COUNT-2:            arith.select
    # CHECK-COUNT-2:            arith.addf
    # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                amdgpu.lds_barrier
    # CHECK-NOT:            amdgpu.lds_barrier
    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
    # CHECK:                scf.for
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
    # CHECK-COUNT-32:           memref.load %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-2:            arith.cmpi slt
    # CHECK-COUNT-2:            arith.select
    # CHECK-COUNT-2:            arith.addf
    # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-4:       vector.maskedstore


@run_test
def test_causal_extend_attention():
    shape = AttentionShape(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        num_seqs=2,
        max_seq_len=32,
        block_size=64,
    )
    total_token_num = 12189
    extend_token_num = 3198
    logit_cap = 30.0
    q_shape = (extend_token_num, shape.num_query_heads, shape.head_size)
    k_shape = (extend_token_num, shape.num_kv_heads, shape.head_size)
    v_shape = (extend_token_num, shape.num_kv_heads, shape.head_size_kv)
    o_shape = (extend_token_num, shape.num_query_heads, shape.head_size_kv)
    k_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    v_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        is_causal=True,
        logit_cap=logit_cap,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
    )
    extend_attention = wave_compile(options, extend_attention)
    print(extend_attention.asm)

    # CHECK-LABEL:       test_causal_extend_attention
    # CHECK-DAG:            #[[map32:.*]] = affine_map<()[s0] -> (s0 * 64 + 64)>
    # CHECK-LABEL:       func.func @extend_attention
    # CHECK-DAG:            %[[workgroup_id_0:.*]] = gpu.block_id x
    # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
    # CHECK-DAG:            %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            %[[ALLOC0:.*]] = memref.alloc() : memref<8704xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC1:.*]] = memref.view %[[ALLOC0]][%[[C0]]][] : memref<8704xi8, #gpu.address_space<workgroup>> to memref<32x1x68xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC2:.*]] = memref.view %[[ALLOC0]][%[[C4352]]][] : memref<8704xi8, #gpu.address_space<workgroup>> to memref<1x32x68xf16, #gpu.address_space<workgroup>>
    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                scf.for
    # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
    # CHECK-COUNT-3:            vector.maskedload
    # CHECK-COUNT-2:            vector.maskedload
    # CHECK-NEXT:               vector.store %{{.*}}, %[[ALLOC2]]
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
    # CHECK-COUNT-32:           memref.load %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
    # CHECK-COUNT-8:            amdgpu.mfma

    # softcap/logitcap modifier:
    # CHECK-COUNT-4:            arith.mulf

    # Tanh Approximation
    # CHECK:                    math.absf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               math.exp2
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               arith.divf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               math.copysign

    # Apply softcap/logitcap modifier
    # CHECK-COUNT-2:            arith.mulf

    # unaligned attention masking:
    # CHECK-COUNT-2:            arith.cmpi slt
    # CHECK-COUNT-2:            arith.select
    # CHECK-COUNT-2:            arith.addf

    # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma

    # Expressions to compute loop bound based on causal mask
    # CHECK:                %[[NQ_TILE_UPPER_BOUND:.*]] = affine.apply #[[map32]]()[%[[workgroup_id_0]]]
    # CHECK:                arith.minsi %[[NQ_TILE_UPPER_BOUND]], {{.*}} : index

    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                amdgpu.lds_barrier
    # CHECK-NOT:            amdgpu.lds_barrier

    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>

    # CHECK:                scf.for
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
    # CHECK-COUNT-32:           memref.load %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
    # CHECK-COUNT-8:            amdgpu.mfma

    # softcap/logitcap modifier:
    # CHECK-COUNT-4:            arith.mulf

    # Tanh Approximation
    # CHECK:                    math.absf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               math.exp2
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               arith.divf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               math.copysign

    # Apply softcap/logitcap modifier
    # CHECK-COUNT-2:            arith.mulf

    # unaligned and causal masking:
    # CHECK-COUNT-1:            arith.cmpi slt
    # CHECK-COUNT-2:            arith.cmpi sge
    # CHECK-COUNT-2:            arith.andi
    # CHECK-COUNT-2:            arith.select
    # CHECK-COUNT-2:            arith.addf

    # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-4:       vector.maskedstore


@run_test
def test_causal_extend_attention_32x32x8():
    shape = AttentionShape(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        num_seqs=2,
        max_seq_len=32,
        block_size=64,
    )
    total_token_num = 12189
    extend_token_num = 3198
    logit_cap = 30.0
    q_shape = (extend_token_num, shape.num_query_heads, shape.head_size)
    k_shape = (extend_token_num, shape.num_kv_heads, shape.head_size)
    v_shape = (extend_token_num, shape.num_kv_heads, shape.head_size_kv)
    o_shape = (extend_token_num, shape.num_query_heads, shape.head_size_kv)
    k_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    v_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    mfma_variant = (tkw.MMAType.F32_32x32x8_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        is_causal=True,
        logit_cap=logit_cap,
        num_waves=2,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
        minimize_shared_allocs=True,
    )
    extend_attention = wave_compile(options, extend_attention)
    print(extend_attention.asm)

    # CHECK-LABEL:       test_causal_extend_attention_32x32x8
    # CHECK:             func.func @extend_attention
    # CHECK-DAG:            %[[C4608:.*]] = arith.constant 4608 : index
    # CHECK-DAG:            %[[C4352:.*]] = arith.constant 4352 : index
    # CHECK-DAG:            %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
    # CHECK-DAG:            %[[ALLOC0:.*]] = memref.alloc() : memref<8960xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC1:.*]] = memref.view %[[ALLOC0]][%[[C0]]][] : memref<8960xi8, #gpu.address_space<workgroup>> to memref<32x1x68xf16, #gpu.address_space<workgroup>>
    # CHECK-DAG:            %[[ALLOC2:.*]] = memref.view %[[ALLOC0]][%[[C4352]]][] : memref<8960xi8, #gpu.address_space<workgroup>> to memref<1x32x68xf16, #gpu.address_space<workgroup>>
    # CHECK-COUNT-8:        vector.maskedload
    # CHECK:                scf.for
    # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
    # CHECK-COUNT-3:            vector.maskedload
    # CHECK-COUNT-2:            vector.maskedload
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
    # CHECK-COUNT-32:           memref.load %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
    # CHECK-COUNT-8:            amdgpu.mfma

    # softcap/logitcap modifier:
    # CHECK-COUNT-2:            arith.mulf

    # Tanh Approximation
    # CHECK:                    math.absf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               math.exp2
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               arith.divf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               math.copysign

    # Apply softcap/logitcap modifier
    # CHECK-COUNT-1:            arith.mulf

    # unaligned attention masking:
    # CHECK-COUNT-1:            arith.cmpi slt
    # CHECK-COUNT-1:            arith.select
    # CHECK-COUNT-1:            arith.addf

    # CHECK-COUNT-2:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK:                %[[ALLOC3:.*]] = memref.view %[[ALLOC0]][%[[C0]]][] : memref<8960xi8, #gpu.address_space<workgroup>> to memref<1x64x36xf16, #gpu.address_space<workgroup>>
    # CHECK:                %[[ALLOC4:.*]] = memref.view %[[ALLOC0]][%[[C4608]]][] : memref<8960xi8, #gpu.address_space<workgroup>> to memref<1x32x68xf16, #gpu.address_space<workgroup>>
    # CHECK-COUNT-8:        vector.maskedload
    # CHECK:                amdgpu.lds_barrier
    # CHECK-NOT:            amdgpu.lds_barrier
    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
    # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
    # CHECK:                scf.for
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-2:            vector.store %{{.*}}, %[[ALLOC4]]
    # CHECK-COUNT-8:            vector.maskedload %{{.*}}
    # CHECK-COUNT-8:            vector.load %[[ALLOC4]]
    # CHECK-COUNT-8:            amdgpu.mfma

    # softcap/logitcap modifier:
    # CHECK-COUNT-2:            arith.mulf

    # Tanh Approximation
    # CHECK:                    math.absf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               math.exp2
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               arith.divf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.mulf
    # CHECK-NEXT:               arith.addf
    # CHECK-NEXT:               math.copysign

    # Apply softcap/logitcap modifier
    # CHECK-COUNT-1:            arith.mulf

    # unaligned and causal masking:
    # CHECK-COUNT-1:            arith.cmpi slt
    # CHECK-COUNT-1:            arith.cmpi sge
    # CHECK-COUNT-1:            arith.andi
    # CHECK-COUNT-1:            arith.select
    # CHECK-COUNT-1:            arith.addf

    # CHECK-COUNT-2:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            amdgpu.mfma
    # CHECK-COUNT-2:       vector.maskedstore


@run_test
def test_extend_attention_custom_mask():
    shape = AttentionShape(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        num_seqs=2,
        max_seq_len=32,
        block_size=64,
    )
    total_token_num = 12189
    extend_token_num = 3198
    q_shape = (extend_token_num, shape.num_query_heads, shape.head_size)
    k_shape = (extend_token_num, shape.num_kv_heads, shape.head_size)
    v_shape = (extend_token_num, shape.num_kv_heads, shape.head_size_kv)
    o_shape = (extend_token_num, shape.num_query_heads, shape.head_size_kv)
    k_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    v_cache_shape = (total_token_num, shape.num_kv_heads, shape.head_size)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        use_custom_mask=True,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        compile_to_mlir=True,
        minimize_shared_allocs=True,
    )
    extend_attention = wave_compile(options, extend_attention)
    print(extend_attention.asm)

    # CHECK-LABEL:       test_extend_attention_custom_mask
    # CHECK-DAG:            #[[map34:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s3 * 32 + s4 + s5 + (s0 + s1 * 64 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16) * s2 + ((s0 mod 64) floordiv 16) * 4)>
    # CHECK-DAG:            #[[map35:.*]] = affine_map<()[s0, s1, s2, s3, s4, s5] -> (s3 * 32 + s4 + s5 + (s0 + s1 * 64 - (s0 floordiv 16) * 16 + (s0 floordiv 64) * 16) * s2 + ((s0 mod 64) floordiv 16) * 4 + 16)>
    # CHECK-LABEL:       func.func @extend_attention_custom_mask
    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                scf.for
    # load and apply custom mask
    # CHECK:                    vector.maskedload
    # CHECK:                    arith.trunci %{{.*}} : vector<4xi8> to vector<4xi1>
    # CHECK:                    arith.andi %{{.*}}, %{{.*}} : vector<4xi1>
    # CHECK-COUNT-8:            amdgpu.mfma

    # CHECK:                scf.for
    # load and apply custom mask
    # CHECK:                    vector.maskedload
    # CHECK:                    arith.trunci %{{.*}} : vector<4xi8> to vector<4xi1>
    # CHECK:                    arith.andi %{{.*}}, %{{.*}} : vector<4xi1>
    # CHECK-COUNT-8:            amdgpu.mfma
