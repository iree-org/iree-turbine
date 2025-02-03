# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import functional as F
from torch.testing import assert_close
from typing import Any, Callable

from iree.turbine.kernel.gen import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel as get_vanilla_tkw_attention_kernel,
)
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config,
    to_default_device,
)
from attention_alibi_template import get_alibi_attention_kernel

torch.manual_seed(0)
torch.set_printoptions(
    linewidth=1000000,
    threshold=1000000,
    precision=3,
)


### TKW Harness
def run(fun: Callable, hparams, *args) -> Any:
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():  # Disable gradient calculations
        with TestLaunchContext(
            hparams,
            canonicalize=True,
            run=False,
            run_config=get_default_run_config(),
            run_bench=False,
            schedule=False,
            use_scheduling_barriers=False,
        ):
            fun(*args)

    # print(
    #     prof.key_averages(group_by_input_shape=True).table(
    #         sort_by="self_cuda_time_total", row_limit=10))


#################################################################################
# INIT VALS
#################################################################################
# num_query_heads, num_kv_heads, head_size, head_size_kv
shape = AttentionShape(128, 128, 128, 128)
shape.query_seq_len = 128
shape.kv_seq_len = 128

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
tkw_attention_alibi_output = device_zeros(o_shape, dtype=torch.float32)

log2e = 1.44269504089
dk_sqrt = math.sqrt(1.0 / q.shape[-1])


#################################################################################
# ALIBI INIT VALS
#################################################################################


def get_relative_positions(seq_len: int) -> torch.Tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return torch.minimum(x - y, torch.zeros(seq_len, seq_len))


def precompute_alibi_slopes(n_heads: int) -> torch.Tensor:
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return to_default_device(m)


alibi_slopes = precompute_alibi_slopes(shape.num_query_heads)


#################################################################################
# TORCH ATTENTION and ATTENTION + ALiBi
#################################################################################
torch_attention_ref_output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None
)

a = torch.matmul(q, k.transpose(-1, -2)) * dk_sqrt
torch_attention_output = torch.matmul(torch.softmax(a, dim=-1), v)

# Sanity check that torch_attention_output and torch_attention_ref_output are
# the same so we can inject ALiBi pre-softmax and compute the delta.
# We will test that the delta post-softmax is the same for torch and TKW.
assert_close(torch_attention_output, torch_attention_ref_output, atol=2e-3, rtol=2e-3)

a += alibi_slopes.unsqueeze(-1).unsqueeze(-1) * get_relative_positions(
    q_shape[1]
).unsqueeze(0)
torch_attention_alibi_output = torch.matmul(F.softmax(a, dim=-1), v)
torch_alibi_delta_output = torch_attention_alibi_output - torch_attention_output


#################################################################################
# TKW BASE ATTENTION
#################################################################################
### Reference version
tkw_attention, hyperparams, dynamic_symbols, dynamic_symbols_map = (
    get_vanilla_tkw_attention_kernel(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16],
        dynamic_dims=False,
    )
)


def attention(tq, tk, tv, toutput):
    tkw_attention(tq, tk, tv, toutput)


run(
    attention,
    hyperparams,
    q * dk_sqrt * log2e,
    k,
    v.permute([0, 2, 1]),
    tkw_attention_output,
)

assert_close(
    torch_attention_output.to(dtype=tkw_attention_output.dtype),
    tkw_attention_output,
    atol=2e-3,
    rtol=2e-3,
)

### ALiBi version
tkw_alibi_attention, hyperparams, dynamic_symbols, dynamic_symbols_map = (
    get_alibi_attention_kernel(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16],
        dynamic_dims=False,
        max_context_length=max_context_length + 2,
    )
)


def attention_alibi(tq, tk, tv, trpe, alibi_slopes):
    mb = tkw_alibi_attention(tq, tk, tv, trpe, alibi_slopes)
    print(mb.module_op)


run(
    attention_alibi,
    hyperparams,
    q * dk_sqrt * log2e,
    k,
    v.permute([0, 2, 1]),
    rpe,
    tkw_attention_alibi_output,
    alibi_slopes,
)

tkw_rpe_delta_output = tkw_attention_alibi_output - tkw_attention_output

assert_close(
    torch_alibi_delta_output.to(dtype=tkw_rpe_delta_output.dtype),
    tkw_rpe_delta_output,
    atol=2e-3,
    rtol=2e-3,
)
