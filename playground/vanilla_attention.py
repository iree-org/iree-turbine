# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from iree.turbine.kernel.gen import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from iree.turbine.kernel.wave.utils import (device_randn, device_zeros,
                                            get_default_run_config)
from vanilla_attention_template import get_vanilla_attention_kernel

# num_query_heads, num_kv_heads, head_size, head_size_kv
shape = AttentionShape(1, 128, 8, 64)
shape.query_seq_len = 64
shape.kv_seq_len = 64
base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map = \
    get_vanilla_attention_kernel(
        shape,
        mfma_variant=[MMAType.F32_16x16x16_F16,
                      MMAType.F32_16x16x16_F16],
        dynamic_dims=False)

vB = shape.num_query_heads
vM = int(shape.query_seq_len)
vN = shape.head_size_kv
vK1 = shape.head_size
vK2 = int(shape.kv_seq_len)
# Override manually to run.
with TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_config=get_default_run_config(),
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
):
    torch.manual_seed(0)
    vB = shape.num_query_heads
    vM = int(shape.query_seq_len)
    vN = shape.head_size_kv
    vK1 = shape.head_size
    vK2 = int(shape.kv_seq_len)
    q = device_randn(vB, vM, vK1, dtype=torch.float16)
    k = device_randn(vB, vK2, vK1, dtype=torch.float16)
    v = device_randn(vB, vN, vK2, dtype=torch.float16)
    # Applied pre-softmax on the MMA'ed result so f32.
    output = device_zeros(vB, vM, vN, dtype=torch.float32)
    # Print IR if needed.
    base_attention(q, k, v, output)
    print(output)
