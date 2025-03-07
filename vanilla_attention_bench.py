# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import torch
from torch.nn import functional as F
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from tests.kernel.wave.common.utils import (
    enable_scheduling_barriers,
)

intrinsics_list = [
        (MMAType.F32_32x32x16_K8_F16, MMAType.F32_32x32x8_F16),
        (MMAType.F32_16x16x32_K8_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ]

def testAttentionPure(
    input_shape: tuple[int],
    enable_scheduling: bool,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
):
    run_bench = True
    dump_perf = True
    shape = AttentionShape(
        num_query_heads=input_shape[0],
        num_kv_heads=input_shape[0],
        query_seq_len=input_shape[1],
        head_size_kv=input_shape[2],
        head_size=input_shape[3],
        kv_seq_len=input_shape[4],
    )
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(shape, mfma_variant, dynamic_dims)
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 100
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = str(input_shape)[1:-1].replace(", ", "x") + "_mfma_" + str(INTRINSIC_ID) + "_dyn-dims_" + str(dynamic_dims) + ".json"
        config["benchmark_results_file"] = os.path.join("tk_" + perf_filename
        )
    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        compile_config=compile_config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
        torch.manual_seed(0)
        q = device_randn(q_shape, dtype=torch.float16)
        k = device_randn(k_shape, dtype=torch.float16)
        v = device_randn(v_shape, dtype=torch.float16)
        output = device_zeros(o_shape, dtype=torch.float32)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape.head_size)
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        mb = base_attention(q * dk_sqrt * log2e, k, v.permute([0, 2, 1]), output)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None
        )

if __name__ == "__main__":
    attention_shapes_list = [(32,1024,128,128,1024), (48,1024,128,128,1024), (32,1024,128,128,1357), (48,1024,128,128,1357)]
    for shape in attention_shapes_list:
        for INTRINSIC_ID in [0,1,2,3]:
            testAttentionPure(shape, False, True, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, True, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, True, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, True, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, False, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, False, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, False, intrinsics_list[INTRINSIC_ID])
            testAttentionPure(shape, False, False, intrinsics_list[INTRINSIC_ID])
