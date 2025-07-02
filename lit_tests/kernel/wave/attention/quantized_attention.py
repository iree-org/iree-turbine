# RUN: python %s | FileCheck %s

import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile


@run_test
def test_fp8_pertensor_attention():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x32_F8, tkw.MMAType.F32_16x16x32_K4_F8)
    base_attention, hyperparams, _ = get_brevitas_pertensor_fp8_attention_kernel(
        shape,
        mfma_variant,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention
    # CHECK:                %[[F8_MAX:.+]] = arith.constant dense<2.400000e+02> : vector<4xf32>
    # CHECK:                %[[F8_OFFSET:.+]] = arith.constant dense<7.90689039> : vector<4xf32>
    # CHECK:                %[[FUSED_SCALE:.+]] = arith.constant dense<0.180336878> : vector<4xf32>
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:            {{.*}} = arith.cmpi slt
    # CHECK-COUNT-4:            {{.*}} = arith.select
    # CHECK-COUNT-8:            {{.*}} = arith.addf

    # fused QK scaling + dequant scaling
    # CHECK-COUNT-8:            {{.*}} = arith.mulf %{{.*}}, %[[FUSED_SCALE]] : vector<4xf32>

    # CHECK-COUNT-4:            {{.*}} = gpu.shuffle xor {{.*}}

    # FP8 Offset
    # CHECK-COUNT-8:            {{.*}} = arith.addf %{{.*}}, %[[F8_OFFSET]] : vector<4xf32>

    # CHECK-COUNT-4:            {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            {{.*}} = arith.minimumf %{{.*}}, %[[F8_MAX]] : vector<4xf32>
    # CHECK-COUNT-8:            {{.*}} = arith.truncf %{{.+}}: vector<4xf32> to vector<4xf8E4M3FNUZ>
    # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
