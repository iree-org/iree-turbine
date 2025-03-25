# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
    paged_decode_attention_shape,
)
from iree.turbine.kernel.wave.templates.decode_attention import (
    get_decode_attention_kernels,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import torch


@run_test
def test_paged_flash_decoding():
    shape = paged_decode_attention_shape(
        num_query_heads=128,
        num_kv_heads=4,
        head_size=32,
        head_size_kv=32,
        block_size=64,
        num_seqs=8,
        kv_lens=100,
    )
    num_kv_splits = 8
    k_shape = (shape.num_seqs, shape.kv_lens, shape.num_kv_heads, shape.head_size)
    v_shape = (shape.num_seqs, shape.kv_lens, shape.num_kv_heads, shape.head_size_kv)
    q_shape = (shape.num_seqs, shape.num_query_heads, shape.head_size)
    o_shape = (shape.num_seqs, shape.num_query_heads, shape.head_size_kv)
    logits_shape = (num_kv_splits, shape.num_seqs, shape.head_size_kv, shape.kv_lens)
    logits_max_shape = (num_kv_splits, shape.num_seqs, shape.head_size_kv)
    block_table_shape = (shape.num_seqs, shape.kv_lens)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16, tkw.MMAType.F32_16x16x16_F16)
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
    ) = get_paged_decode_attention_kernels(
        shape, mfma_variant, num_kv_splits, k_shape, v_shape, block_table_shape
    )

    torch.manual_seed(0)
    q = torch.randn(q_shape, dtype=torch.float16)
    k = torch.randn(k_shape, dtype=torch.float16)
    v = torch.randn(v_shape, dtype=torch.float16)
    logits = torch.zeros(logits_shape, dtype=torch.float32)
    logits_max = torch.zeros(logits_max_shape, dtype=torch.float32)
    output = torch.zeros(o_shape, dtype=torch.float16)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
    ):
        print(phase_0(q, k, v, logits, logits_max))

    # CHECK-LABEL:          func.func @phase_0
    # CHECK-DAG:               %[[C0:.*]] = arith.constant 0 : index
    # CHECK-DAG:               %[[C1:.*]] = arith.constant 1 : index
    # CHECK-DAG:               %[[C16:.*]] = arith.constant 16 : index
    # CHECK-COUNT-2:           vector.load
    # CHECK:                   %[[COUNT0:.*]] = arith.minsi %{{.*}}, %{{.*}} : vector<1xindex>
    # CHECK:                   %[[COUNT1:.*]] = vector.extract %[[COUNT0]][0] : index from vector<1xindex>
    # CHECK-COUNT-2:           vector.load
    # CHECK:                   %[[D36:.*]] = arith.subi %[[COUNT1]], %[[C1]] : index
    # CHECK:                   %[[D37:.*]] = arith.divui %[[D36]], %[[C16]] : index
    # CHECK:                   %[[D38:.*]] = arith.addi %[[D37]], %[[C1]] : index
    # CHECK:                   %[[COUNT2:.*]] = arith.select %{{.*}}, %[[C0]], %[[D38]] : index
    # CHECK:                   scf.for %{{.*}} = %[[C0]] to %[[COUNT2]] step %[[C1]]
    # CHECK:                        amdgpu.lds_barrier
    # 1 masked load block table, 1 for k_cache, and 1 for v_cache.
    # CHECK-COUNT-3:                vector.maskedload
    # CHECK:                        amdgpu.lds_barrier
    # CHECK-COUNT-4:                amdgpu.mfma
    # CHECK:                  %[[COND:.*]] = arith.cmpi sgt, %[[COUNT1]], %[[C0]] : index
    # CHECK:                  scf.if %[[COND]] {
    # CHECK-COUNT-1:          arith.divf
    # CHECK-COUNT-1:          math.log2
    # CHECK-COUNT-9:         vector.store

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
    ):
        print(phase_1(logits, logits_max, output))

    # CHECK-LABEL:       func.func @phase_1
    # CHECK:               vector.load
    # CHECK:               scf.for
    # CHECK-COUNT-2:           vector.maskedload
    # CHECK-COUNT-1:           arith.maximumf
    # CHECK-COUNT-1:           arith.subf
    # CHECK-COUNT-1:           math.exp2
    # CHECK-COUNT-1:           arith.subf
    # CHECK-COUNT-1:           math.exp2
    # CHECK-COUNT-1:           arith.mulf
    # CHECK-COUNT-1:           arith.addf
    # CHECK-COUNT-2:           arith.mulf
    # CHECK-COUNT-1:           arith.addf
    # CHECK-COUNT-1:      arith.divf
    # CHECK-COUNT-1:      vector.store


@run_test
def test_flash_decoding():
    shape = (8, 128, 128, 64, 256)
    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    use_dynamic_dims = True
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_map_0,
        dynamic_symbols_1,
        dynamic_symbols_map_1,
    ) = get_decode_attention_kernels(shape, mfma_variant, use_dynamic_dims)

    torch.manual_seed(0)
    q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    logits = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    logits_max = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols_0,
        dynamic_symbols_map=dynamic_symbols_map_0,
    ):
        print(phase_0(q, k, v, logits, logits_max))

    # CHECK-LABEL:           func.func @phase_0
    # CHECK-NOT:               {{.*}} = scf.for
    # CHECK-COUNT-9:           {{.*}} = vector.maskedload
    # CHECK-COUNT-1:           vector.store
    # CHECK-COUNT-4:           {{.*}} = vector.load
    # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:           {{.*}} = gpu.shuffle
    # CHECK-COUNT-2:           {{.*}} = arith.subf
    # CHECK-COUNT-2:           {{.*}} = math.exp2
    # CHECK-COUNT-2:           {{.*}} = arith.subf
    # CHECK-COUNT-2:           {{.*}} = math.exp2
    # CHECK-COUNT-4:           {{.*}} = gpu.shuffle
    # CHECK-COUNT-2:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-2:           {{.*}} = arith.divf
    # CHECK-COUNT-2:           {{.*}} = math.log2
    # CHECK-COUNT-18:          vector.maskedstore

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols_1,
        dynamic_symbols_map=dynamic_symbols_map_1,
    ):
        print(phase_1(logits, logits_max, output))

    # CHECK-LABEL:       func.func @phase_1
    # CHECK:               {{.*}} = scf.for
    # CHECK-COUNT-2:           {{.*}} = vector.maskedload
    # CHECK-COUNT-1:           {{.*}} = arith.maximumf
    # CHECK-COUNT-1:           {{.*}} = arith.subf
    # CHECK-COUNT-1:           {{.*}} = math.exp2
    # CHECK-COUNT-1:           {{.*}} = arith.subf
    # CHECK-COUNT-1:           {{.*}} = math.exp2
    # CHECK-COUNT-1:           {{.*}} = arith.mulf
    # CHECK-COUNT-1:           {{.*}} = arith.addf
    # CHECK-COUNT-2:           {{.*}} = arith.mulf
    # CHECK-COUNT-1:           {{.*}} = arith.addf
    # CHECK-COUNT-1:      {{.*}} = arith.divf
    # TODO: Remove vector.scatter when optimizing for performance
    # CHECK-COUNT-1:      vector.scatter
