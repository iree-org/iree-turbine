# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import functional as F
from typing import Any, Callable

from iree.turbine.kernel.gen import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel as get_vanilla_attention_kernel_reference)
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config,
    to_default_device,
)
from vanilla_attention_template import get_vanilla_attention_kernel

torch.manual_seed(0)
torch.set_printoptions(
    linewidth=10000,
    threshold=10000,
    precision=1,
)

# num_query_heads, num_kv_heads, head_size, head_size_kv
shape = AttentionShape(1, 128, 8, 64)
shape.query_seq_len = 64
shape.kv_seq_len = 64

# T5 RPE parameter
max_context_length = 24

base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map = \
    get_vanilla_attention_kernel(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16,
                      MMAType.F32_16x16x16_F16],
        dynamic_dims=False,
        max_context_length = max_context_length + 2)

base_attention_reference, _, _, _ = \
    get_vanilla_attention_kernel_reference(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16,
                      MMAType.F32_16x16x16_F16],
        dynamic_dims=False)

vB = shape.num_query_heads
vM = int(shape.query_seq_len)
vN = shape.head_size_kv
vK1 = shape.head_size
vK2 = int(shape.kv_seq_len)
q = device_randn(vB, vM, vK1, dtype=torch.float16)
k = device_randn(vB, vK2, vK1, dtype=torch.float16)
v = device_randn(vB, vN, vK2, dtype=torch.float16)
output = device_zeros(vB, vM, vN, dtype=torch.float32)
output_reference = device_zeros(vB, vM, vN, dtype=torch.float32)
output_reference_2 = device_zeros(vB, vM, vN, dtype=torch.float32)

# Applied pre-softmax on the MMA'ed result so f32.
# Provision more room for clipping and adding 0 at the boundaries.
rpe = device_randn(max_context_length + 2, dtype=torch.float32)
rpe[0] = 0
rpe[max_context_length + 1] = 0


def run(fun: Callable, hparams, *args) -> Any:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():  # Disable gradient calculations
            with TestLaunchContext(
                    hparams,
                    canonicalize=True,
                    run=True,
                    run_config=get_default_run_config(),
                    run_bench=False,
                    schedule=False,
                    use_scheduling_barriers=False,
            ):
                fun(*args)

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=10))


def attention_with_rpe(tq, tk, tv, trpe, toutput):
    # Print IR if needed.
    # print(base_attention(q, k, v, rpe, output).module_op)
    base_attention(tq, tk, tv, trpe, toutput)


def attention_reference(tq, tk, tv, toutput):
    base_attention_reference(tq, tk, tv, toutput)


# run(attention_with_rpe, hyperparams, q, k, v, rpe, output)
# run(attention_reference, hyperparams, q, k, v, output_reference)
# run(attention_reference, hyperparams, q, k, v, output_reference_2)
# print(f"\n\nreference:\n{output_reference.cpu()[0]}")
# print(f"RPE:\n{rpe.cpu()}")
# print(f"ATTENTION RPE:\n{output.cpu()[0]}")
# print(f"delta:\n{(output - output_reference).cpu()[0]}")
# print(
#     f"truth sanity check should be zero:\n{(output_reference - output_reference_2).cpu()[0]}"
# )


def t5_rpe_masked_cond(rpe, max_context_length: int, sequence_length: int,
                       dtype):
    positions = to_default_device(torch.arange(sequence_length))
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = to_default_device((pos_diff >= 0) & (pos_diff < max_context_length))
    rpe_cond = device_zeros(sequence_length, sequence_length, dtype=dtype)
    rpe_cond[mask] = rpe[pos_diff[mask]]
    return rpe_cond


log2e = 1.44269504089
dk_sqrt = math.sqrt(1.0 / shape.kv_seq_len)
a = torch.matmul(q, k.transpose(-1, -2)) * dk_sqrt
torch_ref = torch.matmul(F.softmax(a, dim=-1), v)

rpe_cond = t5_rpe_masked_cond(
    rpe,
    max_context_length=max_context_length,
    sequence_length=shape.kv_seq_len,
    dtype=output.dtype)
a += rpe_cond.unsqueeze(0)
torch_ref2 = torch.matmul(F.softmax(a, dim=-1), v)

# Should be a lower multi-diagonal of width max_context_length starting at (1, 0)
# print(rpe_cond)
# print(torch_ref2 - torch_ref)
