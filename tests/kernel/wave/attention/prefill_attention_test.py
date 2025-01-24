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
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_scheduling_params,
    device_arange,
    device_full,
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.prefill_attention import (
    get_prefill_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
)
import os
from torch.testing import assert_close
from ..common.utils import (
    require_e2e,
    require_cdna3,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from ..common.shapes import get_test_shapes
from typing import List, Optional

# Reference paged attention implementation from vLLM and sglang.
# (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, HEAD_SIZE_KV, SEQ_LENS)
shapes = [(4, 1, 64, 64, (128, 256))]


# From: https://github.com/sgl-project/sglang/blob/main/test/srt/test_triton_attention_kernels.py
def validate_accuracy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: List[int],
    causal: bool,
    output: torch.Tensor,
) -> torch.Tensor:
    cu_seq_lens = [0] * (len(seq_lens) + 1)
    for i, seq_len in enumerate(seq_lens):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(seq_lens)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        o_torch = torch.nn.functional.scaled_dot_product_attention(
            query[start:end].permute(1, 0, 2),
            key[start:end].permute(1, 0, 2),
            value[start:end].permute(1, 0, 2),
            is_causal=causal,
        ).permute(1, 0, 2)
        assert_close(
            output[start:end], o_torch, check_dtype=False, rtol=1e-3, atol=1e-3
        )

    return o_torch


def create_inputs(
    shape: AttentionShape,
    seq_lens: tuple[int],
    dtype: torch.dtype,
):
    query = device_randn(
        shape.total_seq_len, shape.num_query_heads, shape.head_size, dtype=dtype
    )
    key = device_randn(
        shape.total_seq_len, shape.num_kv_heads, shape.head_size, dtype=dtype
    )
    value = device_randn(
        shape.total_seq_len, shape.num_kv_heads, shape.head_size, dtype=dtype
    )
    start_offsets = [0]
    for seq in seq_lens[:-1]:
        start_offsets.append(start_offsets[-1] + seq)
    start_offsets = torch.tensor(start_offsets, device=value.device, dtype=torch.int32)
    seq_lens = torch.tensor(seq_lens, device=value.device, dtype=torch.int32)
    return (query, key, value, start_offsets, seq_lens)


# TODO: Investigate errors on MI250.
@require_e2e
@require_cdna3
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [(MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)],
)
def testPrefillAttention(
    shape: tuple[int],
    dtype: torch.dtype,
    enable_scheduling: bool,
    mfma_variant: MMAType,
    request,
):

    torch.manual_seed(0)
    seq_lens = shape[4]
    shape = AttentionShape(
        num_query_heads=shape[0],
        num_kv_heads=shape[1],
        head_size=shape[2],
        head_size_kv=shape[3],
        num_seqs=len(seq_lens),
        max_seq_len=max(seq_lens),
        total_seq_len=sum(seq_lens),
    )
    assert shape.num_query_heads % shape.num_kv_heads == 0

    (query, key, value, start_offsets, seq_lens) = create_inputs(shape, seq_lens, dtype)

    output_shape = (shape.total_seq_len, shape.num_query_heads, shape.head_size_kv)
    permuted_value = value.permute(1, 2, 0)
    # Run the wave kernel.
    (prefill, hyperparams) = get_prefill_attention_kernel(
        shape,
        mfma_variant,
        query.shape,
        key.shape,
        permuted_value.shape,
        output_shape,
    )

    hyperparams.update(get_default_scheduling_params())
    config = get_default_run_config()
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        # TODO: Add scaling of QK as part of kernel.
        # TODO: Add variant of non-transposed V attention kernel.
        output = device_zeros(output_shape, dtype=torch.float32)
        mb = prefill(
            query * dk_sqrt * log2e,
            key,
            permuted_value,
            start_offsets,
            seq_lens,
            output,
        )

        validate_accuracy(
            query=query,
            key=key,
            value=value,
            seq_lens=seq_lens,
            causal=False,
            output=output,
        )
