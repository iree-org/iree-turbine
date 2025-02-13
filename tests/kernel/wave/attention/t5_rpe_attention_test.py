# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing import assert_close
from typing import Any, Callable

from iree.turbine.kernel.gen import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config,
    to_default_device,
)
from iree.turbine.kernel.wave.templates.t5_rpe_attention import (
    get_t5_rpe_attention_kernel,
)
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)

torch.manual_seed(0)
torch.set_printoptions(
    linewidth=1000000,
    threshold=1000000,
    precision=3,
)


### TKW Harness
def run(fun: Callable, hparams, *args) -> Any:
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


#################################################################################
# INIT VALS
#################################################################################
# num_query_heads, num_kv_heads, head_size, head_size_kv
# WARNING: for now 128^n is the minimal size because attention_with_rpe_template
# does not support padding and as a consequence we have
shape = AttentionShape(256, 256, 256, 256)
shape.query_seq_len = 256
shape.kv_seq_len = 256

assert (
    shape.num_query_heads == shape.num_kv_heads
), "expected query and kv to have the same number of heads!"

q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)

q = device_randn(q_shape, dtype=torch.float16)
k = device_randn(k_shape, dtype=torch.float16)
v = device_randn(v_shape, dtype=torch.float16)
tkw_attention_output = device_zeros(o_shape, dtype=torch.float32)
tkw_attention_with_rpe_output = device_zeros(o_shape, dtype=torch.float32)

log2e = 1.44269504089
dk_sqrt = math.sqrt(1.0 / q.shape[-1])

#################################################################################
# T5 RPE INIT VALS
#################################################################################
# T5 RPE parameter
max_context_length = 10

# Applied pre-softmax on the MMA'ed result so f32.
# Provision more room for clipping and adding 0 at the boundaries.
rpe = device_zeros(1000 + max_context_length + 2, dtype=torch.float32)
rpe = rpe[: max_context_length + 2].view(max_context_length + 2)
rpe.copy_(device_randn(max_context_length + 2, dtype=torch.float32))
rpe[0] = 0
rpe[max_context_length + 1] = 0


def t5_rpe_masked_cond(rpe, max_context_length: int, sequence_length: int, dtype):
    positions = to_default_device(torch.arange(sequence_length))
    pos_diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    mask = to_default_device((pos_diff >= 0) & (pos_diff <= max_context_length))
    rpe_cond = device_zeros(sequence_length, sequence_length, dtype=dtype)
    rpe_cond[mask] = rpe[pos_diff[mask]]
    return rpe_cond


# rpe_cond is used by torch only and to sanity check the tmp RPE cond that we
# output as debug information.
rpe_cond = t5_rpe_masked_cond(
    rpe,
    max_context_length=max_context_length,
    sequence_length=shape.kv_seq_len,
    dtype=tkw_attention_with_rpe_output.dtype,
)

#################################################################################
# TKW BASE ATTENTION
#################################################################################
### RPE version
(
    tkw_attention_with_rpe,
    hyperparams,
    dynamic_symbols,
    dynamic_symbols_map,
) = get_t5_rpe_attention_kernel(
    shape,
    mfma_variant=[MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16],
    dynamic_dims=False,
    max_context_length=max_context_length + 2,
)


def attention_with_rpe(*args):
    tkw_attention_with_rpe(*args)


run(
    attention_with_rpe,
    hyperparams,
    q * dk_sqrt * log2e,
    k,
    v.permute([0, 2, 1]),
    rpe * log2e,
    tkw_attention_with_rpe_output,
)

### Reference version
(
    tkw_attention_without_rpe,
    hyperparams,
    dynamic_symbols,
    dynamic_symbols_map,
) = get_vanilla_attention_kernel(
    shape,
    mfma_variant=[MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16],
    dynamic_dims=False,
)


def attention(tq, tk, tv, toutput):
    tkw_attention_without_rpe(tq, tk, tv, toutput)


run(
    attention,
    hyperparams,
    q * dk_sqrt * log2e,
    k,
    v.permute([0, 2, 1]),
    tkw_attention_output,
)

#################################################################################
# TORCH ATTENTION and ATTENTION + RPE
#################################################################################
a = torch.matmul(q, k.transpose(-1, -2)) * dk_sqrt
torch_attention_output = torch.matmul(torch.softmax(a, dim=-1), v)
a += rpe_cond.unsqueeze(0)
torch_attention_with_rpe_output = torch.matmul(F.softmax(a, dim=-1), v)

# Check basic attentions match as we expect.
assert_close(
    torch_attention_output.to(dtype=tkw_attention_output.dtype),
    tkw_attention_output,
    atol=2e-3,
    rtol=2e-3,
)

# Check RPE attentions match as we expect.
assert_close(
    torch_attention_with_rpe_output.to(dtype=tkw_attention_with_rpe_output.dtype),
    tkw_attention_with_rpe_output,
    atol=2e-3,
    rtol=2e-3,
)
