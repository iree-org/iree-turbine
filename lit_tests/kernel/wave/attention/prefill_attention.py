# RUN: python %s | FileCheck %s

import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile


@run_test
def test_prefill_attention():
    shape = AttentionShape(
        num_query_heads=16,
        num_kv_heads=4,
        head_size=64,
        head_size_kv=64,
        num_seqs=2,
        max_seq_len=12,
        total_seq_len=20,
    )
    q_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size)
    k_shape = (shape.total_seq_len, shape.num_kv_heads, shape.head_size)
    v_shape = (shape.total_seq_len, shape.num_kv_heads, shape.head_size_kv)
    o_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size_kv)
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    prefill_attention, hyperparams = get_prefill_attention_kernel(
        shape, mfma_variant, q_shape, k_shape, v_shape, o_shape
    )
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    prefill_attention = wave_compile(options, prefill_attention)
    print(prefill_attention.asm)

    # CHECK-LABEL:       func.func @prefill_attention
    # CHECK-COUNT-4:        vector.maskedload
    # CHECK:                scf.for
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store
    # CHECK-COUNT-1:            vector.maskedload
    # CHECK-COUNT-1:            vector.store
    # CHECK-COUNT-8:            vector.maskedload
    # CHECK-COUNT-2:            vector.store
    # CHECK-COUNT-32            vector.load
    # CHECK-COUNT-16:           amdgpu.mfma
    # CHECK-COUNT-4:            gpu.shuffle xor {{.*}}
    # CHECK-COUNT-16:           amdgpu.mfma
    # CHECK-COUNT-4:       vector.maskedstore
