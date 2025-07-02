# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.utils.general_utils import run_test
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType


@run_test
def test_attention():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
        func_name="test_vanilla_attention",
        minimize_shared_allocs=True,
        reorder_allocs=False,
    )
    # In this example, by not reordering the allocs, we end up with a 2x
    # saving in total shared memory usage (from 17408 to 8704).
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # We end up with just 1 allocation that is reused.
    # CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
    # CHECK:        %[[alloc:.+]] = memref.alloc() : memref<8704xi8, #gpu.address_space<workgroup>>
    # CHECK-DAG:    %[[VIEW:.+]] = memref.view %[[alloc]][%[[C0]]][] : memref<8704xi8, #gpu.address_space<workgroup>> to memref<1x64x68xf16, #gpu.address_space<workgroup>>
