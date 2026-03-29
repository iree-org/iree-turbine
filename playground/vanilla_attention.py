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

torch.manual_seed(0)
torch.set_printoptions(
    linewidth=10000,
    threshold=10000,
    precision=3,
)

#################################################################################
# INIT VALS
#################################################################################
# num_query_heads, num_kv_heads, head_size, head_size_kv
shape = AttentionShape(1, 1, 8, 32)
shape.query_seq_len = 32
shape.kv_seq_len = 32

assert shape.num_query_heads == shape.num_kv_heads, \
    "expected query and kv to have the same number of heads!"

mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
(
    base_attention,
    hyperparams,
    dynamic_symbols,
    dynamic_symbols_map,
) = get_vanilla_attention_kernel_reference(shape, mfma_variant, dynamic_dims=None)

q_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size)
k_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size)
v_shape = (shape.num_kv_heads, shape.kv_seq_len, shape.head_size_kv)
o_shape = (shape.num_query_heads, shape.query_seq_len, shape.head_size_kv)
config = get_default_run_config()
with TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_config=config,
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
        q, k, v, attn_mask=None)
    print(output, torch_ref)
    torch.testing.assert_close(output, torch_ref, check_dtype=False, atol=1e-3, rtol=1e-3)

