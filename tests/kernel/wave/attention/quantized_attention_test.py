# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    quantized_tensor,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
import os
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
    param_bool,
)
from ..common.shapes import get_test_shapes
from iree.turbine.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions


@require_e2e
@require_cdna3
@pytest.mark.parametrize("input_shape", get_test_shapes("quantized_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_32x32x16_F8, MMAType.F32_32x32x16_K4_F8),
        (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8),
    ],
)
def testAttentionPure(
    input_shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: tuple[MMAType],
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    shape = AttentionShape(
        num_query_heads=input_shape[0],
        num_kv_heads=input_shape[0],
        query_seq_len=input_shape[1],
        head_size_kv=input_shape[2],
        head_size=input_shape[3],
        kv_seq_len=input_shape[4],
    )
    # Sample tensor scaling from Brevitas SDXL-FP8.
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    (
        base_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_brevitas_pertensor_fp8_attention_kernel(
        shape,
        mfma_variant,
    )
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
    hyperparams.update(get_default_scheduling_params())

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        run_bench=run_bench,
        inplace=False,  # we are supporting wave_runtime for scalar codegen for now
        wave_runtime=True,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
    )

    options = set_default_run_config(options)
    base_attention = wave_compile(options, base_attention)

    torch.manual_seed(0)
    # Smaller range to help with FP8 minimum range
    MAX_RANGE = 128
    q = quantized_tensor(q_shape, dtype=torch.float16, scale=MAX_RANGE)
    k = quantized_tensor(k_shape, dtype=torch.float16, scale=MAX_RANGE)
    v = quantized_tensor(v_shape, dtype=torch.float16, scale=MAX_RANGE)

    output = device_zeros(o_shape, dtype=torch.float32)
    asm = base_attention(q, k, v, q_scale, k_scale, v_scale, output)

    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32) * q_scale,
        k.to(torch.float32) * k_scale,
        v.to(torch.float32) * v_scale,
    )

    if dump_generated_mlir:
        filename = f"wave_attention_{'x'.join(map(str, input_shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm)

    rmse = torch.sqrt(torch.mean(torch.square(output - torch_ref)))
    # Higher tolerance because, we are testing a higher range
    # of numbers here (-128, 128), typically device_rand just gets (-1, 1).
    assert rmse < 0.04
