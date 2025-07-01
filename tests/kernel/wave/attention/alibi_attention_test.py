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
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_arange,
    device_full,
    device_randn,
    device_zeros,
    to_default_device,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.alibi_attention import (
    get_alibi_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    enable_scheduling_barriers,
)
from ..common.shapes import get_test_shapes
from typing import List, Optional, Tuple

shapes = [(128, 128, 128, 128, 128, 128)]


def get_relative_positions(
    seq_len: int, kv_seq_len: Optional[int] = None
) -> torch.Tensor:
    """Returns a lower-trinagular tensor with distance between rows and columns.

    The tensor resembles the following:

        [ 0  0  0  0  0]
        [-1  0  0  0  0]
        [-2 -1  0  0  0]
        [-3 -2 -1  0  0]
    """
    if not kv_seq_len:
        kv_seq_len = seq_len
    x = torch.arange(kv_seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return to_default_device(torch.minimum(x - y, torch.zeros(seq_len, kv_seq_len)))


def precompute_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Computes the constant slopes of linear biases to be added to the attention scores."""
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return to_default_device(m)


def validate_accuracy(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    # Precompute values.
    dk_sqrt = math.sqrt(1.0 / query.shape[-1])
    alibi_slopes = precompute_alibi_slopes(query.shape[0])

    # Straightforward implementation of attention with bias.
    scores = torch.matmul(query, key.transpose(-1, -2)) * dk_sqrt
    bias = alibi_slopes.unsqueeze(-1).unsqueeze(-1) * get_relative_positions(
        query.shape[1], key.shape[1]
    )
    bias = bias.to(dtype=scores.dtype)
    scores = scores + bias
    reference = torch.matmul(torch.softmax(scores, dim=-1), value)
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


@require_e2e
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "mfma_variant",
    [(MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)],
)
def test_alibi_attention(
    shape: tuple[int],
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
    alibi_attention, hyperparams, _ = get_alibi_attention_kernel(
        shape, mfma_variant, dynamic_dims=False
    )
    output_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)

    hyperparams.update(get_default_scheduling_params())
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    alibi_slopes = precompute_alibi_slopes(shape.head_size)

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
    alibi_attention = wave_compile(options, alibi_attention)

    output = device_zeros(output_shape, dtype=torch.float32)
    # TODO: Add scaling of QK and ALiBi as part of kernel.
    alibi_attention(
        query * dk_sqrt * log2e,
        key,
        value.permute([0, 2, 1]),
        # NOTE: since the kernel uses exp2 instead of exp, the ALiBi slopes must be
        # multiplied by the same factor as the Q matrix to preserve the result post
        # softmax:  exp(x + alibi) = exp2((x + alibi) * log2(e))
        alibi_slopes * log2e,
        output,
    )

    validate_accuracy(query, key, value, output)
