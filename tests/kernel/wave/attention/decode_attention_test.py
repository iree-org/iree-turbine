# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import math
import iree.turbine.kernel as tk
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.decode_attention import (
    get_decode_attention_kernels,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    param_bool,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("decode_attention"))
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
@param_bool("dynamic_dims", "dyn")
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testFlashDecoding(
    shape: tuple[int],
    enable_scheduling: SchedulingType,
    dynamic_dims: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_map_0,
        dynamic_symbols_1,
        dynamic_symbols_map_1,
    ) = get_decode_attention_kernels(shape, mfma_variant, dynamic_dims)
    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    torch.manual_seed(0)
    B, M, N, K1, K2 = shape
    sym_U = index_symbol("U")
    if sym_U in hyperparams_0:
        U = hyperparams_0[sym_U]
    else:
        U = dynamic_symbols_map_0[sym_U]
    q = device_randn(B, M, K1, dtype=torch.float16)
    k = device_randn(B, K2, K1, dtype=torch.float16)
    v = device_randn(B, K2, N, dtype=torch.float16)
    phase_0_output = device_zeros(U, B, N, M, dtype=torch.float32)
    phase_0_output_max = device_zeros(U, B, M, dtype=torch.float32)
    output = device_zeros(B, M, N, dtype=torch.float32)
    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / K1)

    options = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols_0,
        dynamic_symbols_map=dynamic_symbols_map_0,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_0 = wave_compile(options, phase_0)

    # TODO: Add scaling of QK as part of kernel.
    asm_qk = phase_0(
        q * dk_sqrt * log2e,
        k,
        v.permute([0, 2, 1]),
        phase_0_output,
        phase_0_output_max,
    )

    options = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        run_bench=run_bench,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
        dynamic_symbols=dynamic_symbols_1,
        dynamic_symbols_map=dynamic_symbols_map_1,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    phase_1 = wave_compile(options, phase_1)

    # TODO: Add variant of non-transposed V attention kernel.
    asm_sv = phase_1(phase_0_output, phase_0_output_max, output)

    torch_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None
    )

    if dump_generated_mlir:
        filename = f"wave_phase_0_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_qk)
        filename = f"wave_phase_1_kernel_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_sv)

    assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)
