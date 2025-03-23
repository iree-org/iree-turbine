# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import pytest
import torch
import os

from torch.nn import functional as F
from torch.testing import assert_close

import iree.turbine.kernel as tk
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    to_default_device,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.templates.t5_rpe_attention import (
    get_t5_rpe_attention_kernel,
)
from ..common.shapes import make_shape_param
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
)
from typing import Tuple

shapes = [
    make_shape_param((128, 128, 128, 128, 128, 128), is_perf=False),
    make_shape_param((128, 128, 128, 128, 128, 128), is_perf=True),
]


def t5_rpe_masked_cond(
    rpe: torch.Tensor,
    max_rpe_context_length: int,
    sequence_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = to_default_device(torch.arange(sequence_length))
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = to_default_device((pos_diff >= 0) & (pos_diff <= max_rpe_context_length))
    rpe_cond = device_zeros(sequence_length, sequence_length, dtype=dtype)
    rpe_cond[mask] = rpe[pos_diff[mask]]
    return rpe_cond


def validate_accuracy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rpe: torch.Tensor,
    output: torch.Tensor,
    max_rpe_context_length: int,
) -> torch.Tensor:
    # Precompute values.
    dk_sqrt = math.sqrt(1.0 / query.shape[-1])
    rpe_cond = t5_rpe_masked_cond(
        rpe,
        max_rpe_context_length=max_rpe_context_length,
        sequence_length=key.shape[1],
        dtype=output.dtype,
    )
    a = torch.matmul(query, key.transpose(-1, -2)) * dk_sqrt
    a += rpe_cond.unsqueeze(0)
    reference = torch.matmul(F.softmax(a, dim=-1), value)
    assert_close(reference, output, check_dtype=False, rtol=2e-3, atol=2e-3)
    return reference


def create_inputs(
    shape: AttentionShape, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
    k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
    v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
    q = device_randn(q_shape, dtype=dtype)
    k = device_randn(k_shape, dtype=dtype)
    v = device_randn(v_shape, dtype=dtype)
    return (q, k, v)


# TODO: Debug why failing numerics on MI250.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("max_rpe_context_length", [10, 128])  # T5 RPE parameter
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
        (MMAType.F32_32x32x8_F16, MMAType.F32_32x32x8_F16),
    ],
)
def test_t5_rpe_attention(
    shape: Tuple[int],
    max_rpe_context_length: int,
    dtype: torch.dtype,
    mfma_variant: MMAType,
    request,
):
    torch.manual_seed(0)
    shape = AttentionShape(
        num_query_heads=shape[0],
        num_kv_heads=shape[1],
        head_size=shape[2],
        head_size_kv=shape[3],
        query_seq_len=shape[4],
        kv_seq_len=shape[5],
    )
    assert shape.num_query_heads % shape.num_kv_heads == 0

    (query, key, value) = create_inputs(shape, dtype)
    t5_rpe_attention, hyperparams, _, _ = get_t5_rpe_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
        max_rpe_context_length=max_rpe_context_length,
    )
    output_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)

    hyperparams.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    # Provision more room for clipping and adding 0 at the boundaries.
    rpe = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
    rpe.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
    rpe[max_rpe_context_length] = 0

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_scheduling_barriers=enable_scheduling_barriers,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + request.node.name + ".json")
            if dump_perf
            else None
        ),
    )
    options = set_default_run_config(options)
    t5_rpe_attention = wave_compile(options, t5_rpe_attention)

    output = device_zeros(output_shape, dtype=torch.float32)
    # TODO: Add scaling of QK and t5_rpe as part of kernel.
    t5_rpe_attention(
        query * dk_sqrt * log2e,
        key,
        value.permute([0, 2, 1]),
        # NOTE: since the kernel uses exp2 instead of exp, the t5_rpe slopes must be
        # multiplied by the same factor as the Q matrix to preserve the result post
        # softmax:  exp(x + t5_rpe) = exp2((x + t5_rpe) * log2(e))
        rpe * log2e,
        output,
    )

    validate_accuracy(query, key, value, rpe, output, max_rpe_context_length)
