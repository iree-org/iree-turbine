# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
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
    block_table_shape = (shape.num_seqs, shape.max_seq_len)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        block_table_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
    )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = torch.randn(q_shape, dtype=torch.float16)
        k = torch.randn(k_shape, dtype=torch.float16)
        v = torch.randn(v_shape, dtype=torch.float16)
        output = torch.zeros(o_shape, dtype=torch.float32)
        request_indices = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        start_indices_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        block_table = torch.zeros(block_table_shape, dtype=torch.int32)
        k_cache = torch.zeros(k_cache_shape, dtype=torch.float16)
        v_cache = torch.zeros(v_cache_shape, dtype=torch.float16)
        print(
            extend_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                request_indices,
                sequence_lengths,
                sequence_lengths_extend,
                start_indices_extend,
                output,
            ).module_op
        )
        # This part ensure correctness of WG distribution for extend attention.
        # CHECK:              stream.executable.export public @extend_attention workgroups(%[[ARG0:.+]]: index, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
        # CHECK:                %[[D0:.+]] = arith.subi %[[ARG3]], %c1 : index
        # CHECK:                %[[D1:.+]] = arith.divui %[[D0]], %c64 : index
        # CHECK:                %[[D2:.+]] = arith.addi %[[D1]], %c1 : index
        # CHECK:                %[[D3:.+]] = arith.cmpi eq, %[[ARG3]], %c0 : index
        # CHECK:                %[[NQ_GRID:.+]] = arith.select %[[D3]], %c0, %[[D2]] : index
        # CHECK:                %[[NUM_SEQ:.+]] = arith.muli %[[ARG2]], %c16 overflow<nsw, nuw> : index
        # CHECK:                stream.return %[[NQ_GRID]], %c1, %[[NUM_SEQ]] : index, index, index

        # CHECK-LABEL:       func.func @extend_attention
        # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
        # CHECK-DAG:            %[[ALLOC1:.*]] = memref.alloc() : memref<32x1x68xf16, #gpu.address_space<workgroup>>
        # CHECK-DAG:            %[[ALLOC2:.*]] = memref.alloc() : memref<1x32x68xf16, #gpu.address_space<workgroup>>
        # CHECK-COUNT-4:        vector.maskedload
        # CHECK:                scf.for
        # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
        # CHECK-COUNT-3:            vector.maskedload
        # CHECK-COUNT-2:            vector.maskedload
        # CHECK-NEXT:               vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma
        # CHECK-COUNT-2:            arith.cmpi slt
        # CHECK-COUNT-2:            arith.select
        # CHECK-COUNT-2:            arith.addf
        # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            amdgpu.mfma
        # CHECK:                amdgpu.lds_barrier
        # CHECK-NOT:            amdgpu.lds_barrier
        # CHECK-COUNT-4:        vector.maskedload
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                scf.for
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
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
    block_table_shape = (shape.num_seqs, shape.max_seq_len)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        block_table_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        is_causal=True,
        logit_cap=logit_cap,
    )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = torch.randn(q_shape, dtype=torch.float16)
        k = torch.randn(k_shape, dtype=torch.float16)
        v = torch.randn(v_shape, dtype=torch.float16)
        output = torch.zeros(o_shape, dtype=torch.float32)
        request_indices = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        start_indices_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        block_table = torch.zeros(block_table_shape, dtype=torch.int32)
        k_cache = torch.zeros(k_cache_shape, dtype=torch.float16)
        v_cache = torch.zeros(v_cache_shape, dtype=torch.float16)
        print(
            extend_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                request_indices,
                sequence_lengths,
                sequence_lengths_extend,
                start_indices_extend,
                output,
            ).module_op
        )

        # CHECK-LABEL:       func.func @extend_attention
        # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
        # CHECK-DAG:            %[[ALLOC1:.*]] = memref.alloc() : memref<32x1x68xf16, #gpu.address_space<workgroup>>
        # CHECK-DAG:            %[[ALLOC2:.*]] = memref.alloc() : memref<1x32x68xf16, #gpu.address_space<workgroup>>
        # CHECK-DAG:            %[[NQ_TILE_WG0:.*]] = arith.muli %workgroup_id_0, %[[NQ_TILE_SIZE:.+]]
        # CHECK-COUNT-4:        vector.maskedload
        # CHECK:                scf.for
        # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
        # CHECK-COUNT-3:            vector.maskedload
        # CHECK-COUNT-2:            vector.maskedload
        # CHECK-NEXT:               vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-2:            arith.divf
        # CHECK-COUNT-2:            math.tanh
        # CHECK-COUNT-2:            arith.mulf

        # unaligned attention masking:
        # CHECK-COUNT-2:            arith.cmpi slt
        # CHECK-COUNT-2:            arith.select
        # CHECK-COUNT-2:            arith.addf

        # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            amdgpu.mfma

        # Expressions to compute loop bound based on causal mask
        # CHECK:                %[[NQ_TILE_UPPER_BOUND:.*]] = arith.addi %[[NQ_TILE_WG0]], %[[NQ_TILE_SIZE]]
        # CHECK:                %[[NQ_LOOP_BOUND_SPLAT:.*]] = vector.splat %[[NQ_TILE_UPPER_BOUND]]
        # CHECK:                arith.minsi {{.*}}, %[[NQ_LOOP_BOUND_SPLAT]]

        # CHECK:                amdgpu.lds_barrier
        # CHECK-NOT:            amdgpu.lds_barrier

        # CHECK-COUNT-4:        vector.maskedload
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>

        # CHECK:                scf.for
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-2:            arith.divf
        # CHECK-COUNT-2:            math.tanh
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
    block_table_shape = (shape.num_seqs, shape.max_seq_len)
    mfma_variant = (tkw.MMAType.F32_32x32x8_F16,) * 2
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        block_table_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        is_causal=True,
        logit_cap=logit_cap,
        num_waves=2,
    )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = torch.randn(q_shape, dtype=torch.float16)
        k = torch.randn(k_shape, dtype=torch.float16)
        v = torch.randn(v_shape, dtype=torch.float16)
        output = torch.zeros(o_shape, dtype=torch.float32)
        request_indices = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        start_indices_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        block_table = torch.zeros(block_table_shape, dtype=torch.int32)
        k_cache = torch.zeros(k_cache_shape, dtype=torch.float16)
        v_cache = torch.zeros(v_cache_shape, dtype=torch.float16)
        print(
            extend_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                request_indices,
                sequence_lengths,
                sequence_lengths_extend,
                start_indices_extend,
                output,
            ).module_op
        )

        # CHECK-LABEL:       func.func @extend_attention
        # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
        # CHECK-DAG:            %[[ALLOC1:.*]] = memref.alloc() : memref<32x1x68xf16, #gpu.address_space<workgroup>>
        # CHECK-DAG:            %[[ALLOC2:.*]] = memref.alloc() : memref<1x32x68xf16, #gpu.address_space<workgroup>>
        # CHECK-COUNT-8:        vector.maskedload
        # CHECK:                scf.for
        # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
        # CHECK-COUNT-3:            vector.maskedload
        # CHECK-COUNT-2:            vector.maskedload
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-1:            arith.divf
        # CHECK-COUNT-1:            math.tanh
        # CHECK-COUNT-1:            arith.mulf

        # unaligned attention masking:
        # CHECK-COUNT-1:            arith.cmpi slt
        # CHECK-COUNT-1:            arith.select
        # CHECK-COUNT-1:            arith.addf

        # CHECK-COUNT-2:            gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            amdgpu.mfma
        # CHECK:                amdgpu.lds_barrier
        # CHECK-NOT:            amdgpu.lds_barrier
        # CHECK-COUNT-8:        vector.maskedload
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                scf.for
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-1:            arith.divf
        # CHECK-COUNT-1:            math.tanh
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
def test_causal_extend_attention_v_mfma():
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
    block_table_shape = (shape.num_seqs, shape.max_seq_len)
    mfma_variant = (tkw.MMAType.F32_16x16x32_K8_F16, tkw.MMAType.F32_16x16x16_F16)
    (
        extend_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_shape,
        k_shape,
        v_shape,
        block_table_shape,
        k_cache_shape,
        v_cache_shape,
        o_shape,
        is_causal=True,
        logit_cap=logit_cap,
        num_waves=2,
    )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = torch.randn(q_shape, dtype=torch.float16)
        k = torch.randn(k_shape, dtype=torch.float16)
        v = torch.randn(v_shape, dtype=torch.float16)
        output = torch.zeros(o_shape, dtype=torch.float32)
        request_indices = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths = torch.zeros(shape.num_seqs, dtype=torch.int32)
        sequence_lengths_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        start_indices_extend = torch.zeros(shape.num_seqs, dtype=torch.int32)
        block_table = torch.zeros(block_table_shape, dtype=torch.int32)
        k_cache = torch.zeros(k_cache_shape, dtype=torch.float16)
        v_cache = torch.zeros(v_cache_shape, dtype=torch.float16)
        print(
            extend_attention(
                q,
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                request_indices,
                sequence_lengths,
                sequence_lengths_extend,
                start_indices_extend,
                output,
            ).module_op
        )

        # CHECK-LABEL:       func.func @extend_attention
        # CHECK-DAG:            stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x16x64xf16, strided<[1024, 64, 1], offset: ?>>
        # CHECK-DAG:            %[[ALLOC1:.*]] = memref.alloc() : memref<32x1x68xf16, #gpu.address_space<workgroup>>
        # CHECK-DAG:            %[[ALLOC2:.*]] = memref.alloc() : memref<1x32x68xf16, #gpu.address_space<workgroup>>
        # CHECK-COUNT-8:        vector.maskedload
        # CHECK:                scf.for
        # 3 masked load for sequence idx, 2 for k_cache, and 1 for v_cache.
        # CHECK-COUNT-3:            vector.maskedload
        # CHECK-COUNT-2:            vector.maskedload
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC1]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-1:            arith.divf
        # CHECK-COUNT-1:            math.tanh
        # CHECK-COUNT-1:            arith.mulf

        # unaligned attention masking:
        # CHECK-COUNT-1:            arith.cmpi slt
        # CHECK-COUNT-1:            arith.select
        # CHECK-COUNT-1:            arith.addf

        # CHECK-COUNT-2:            gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            amdgpu.mfma
        # CHECK:                amdgpu.lds_barrier
        # CHECK-NOT:            amdgpu.lds_barrier
        # CHECK-COUNT-8:        vector.maskedload
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                stream.binding.subspan %{{.*}}[%{{.*}}] : !stream.binding -> memref<?x4x64xf16, strided<[256, 64, 1], offset: ?>>
        # CHECK:                scf.for
        # CHECK-COUNT-1:            vector.maskedload
        # CHECK-COUNT-1:            vector.store %{{.*}}, %[[ALLOC2]]
        # CHECK-COUNT-32:           vector.load %{{.*}}
        # CHECK-COUNT-8:            vector.load %[[ALLOC2]]
        # CHECK-COUNT-8:            amdgpu.mfma

        # softcap/logitcap modifier:
        # CHECK-COUNT-1:            arith.divf
        # CHECK-COUNT-1:            math.tanh
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
