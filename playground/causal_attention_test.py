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
from iree.turbine.kernel.wave.utils import (
    device_randn,
    device_zeros,
    get_default_run_config,
    to_default_device,
)
from causal_attention_template import (get_causal_attention_kernel as
                                       get_tkw_causal_attention_kernel)

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
torch.manual_seed(0)
torch.set_printoptions(
    linewidth=1000000,
    threshold=1000000,
    precision=3,
)

### TKW Harness
def run(fun: Callable, hparams, *args) -> Any:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():  # Disable gradient calculations
            with TestLaunchContext(
                    hparams,
                    canonicalize=True,
                    # compile_config={"print_ir_after": "all"},
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


#################################################################################
# INIT VALS
#################################################################################
# num_query_heads, num_kv_heads, head_size, head_size_kv
shape = AttentionShape(128, 128, 128, 128)
shape.query_seq_len = 128
shape.kv_seq_len = 128

assert shape.num_query_heads == shape.num_kv_heads, \
    "expected query and kv to have the same number of heads!"

q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)

q = device_randn(q_shape, dtype=torch.float16)
k = device_randn(k_shape, dtype=torch.float16)
v = device_randn(v_shape, dtype=torch.float16)
tkw_causal_attention_output = device_zeros(o_shape, dtype=torch.float32)

log2e = 1.44269504089
dk_sqrt = math.sqrt(1.0 / q.shape[-1])

#################################################################################
# TORCH ATTENTION
#################################################################################
torch_causal_attention_ref_output = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=None, is_causal=True)

#################################################################################
# TKW ATTENTION
#################################################################################
tkw_causal_attention, hyperparams, dynamic_symbols, dynamic_symbols_map = \
    get_tkw_causal_attention_kernel(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16,
                      MMAType.F32_16x16x16_F16],
        dynamic_dims=False)


def causal_attention(tq, tk, tv, toutput):
    mb = tkw_causal_attention(tq, tk, tv, toutput)
    print(mb.module_op)


run(causal_attention, hyperparams, q * dk_sqrt * log2e, k,
    v.permute([0, 2, 1]), tkw_causal_attention_output)


print(torch_causal_attention_ref_output)
print(tkw_causal_attention_output)


assert_close(torch_causal_attention_ref_output.to(
    dtype=tkw_causal_attention_output.dtype),
             tkw_causal_attention_output,
             atol=2e-3,
             rtol=2e-3)
