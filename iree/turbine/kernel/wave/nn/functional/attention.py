# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import math
import functools
import iree.turbine.kernel.lang as tkl
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape


@functools.lru_cache
def get_wave_kernel(
    shape: AttentionShape,
    is_causal: bool,
    is_v_transposed: bool = False,
    sliding_window_size: int = -1,
):
    assert shape.num_query_heads % shape.num_kv_heads == 0

    mfma_variant = (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16)
    (
        vanilla_attention,
        hyperparams,
        dynamic_symbols,
        dynamic_symbols_map,
    ) = get_vanilla_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
        is_causal=is_causal,
        is_v_transposed=is_v_transposed,
        sliding_window_size=sliding_window_size,
    )
    hyperparams.update(get_default_scheduling_params())
    del hyperparams[tkl.sym.B]
    del hyperparams[tkl.sym.M]
    del hyperparams[tkl.sym.N]
    del hyperparams[tkl.sym.K2]
    dynamic_symbols = [tkl.sym.B, tkl.sym.M, tkl.sym.N, tkl.sym.K2]

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    vanilla_attention = wave_compile(options, vanilla_attention)
    return vanilla_attention


def wave_sdpa(
    query,
    key,
    value,
    is_causal=False,
    is_v_transposed=False,
    sliding_window_size=-1,
):
    """
    SDPA op that handles any dimension of batch as long as SxD is fastest dim.
    i.e layout will be B x <B1 x B2 x .. x B_N> x S x D. This op also expects
    batch dimension to be exactly same across Q,K, and V (i.e no GQA/MQA yet).
    """
    key_shape = key.shape
    query_shape = query.shape
    value_shape = value.shape
    assert key_shape == value_shape
    assert len(key_shape) == len(query_shape)
    assert len(query_shape) >= 2
    assert query_shape[-1] == key_shape[-1]
    assert query_shape[:-2] == key_shape[:-2]
    # Let flattened batch be treated as num_query_heads/num_kv_heads.
    batch = query_shape[:-2]
    flattend_batch_size = math.prod(batch)
    flat_q_shape = [flattend_batch_size, query_shape[-2], query_shape[-1]]
    flat_kv_shape = [flattend_batch_size, key_shape[-2], key_shape[-1]]
    flat_o_shape = [flattend_batch_size, query_shape[-2], key_shape[-1]]
    output = torch.empty(flat_o_shape, dtype=torch.float32, device=query.device)

    shape = AttentionShape(
        num_query_heads=flattend_batch_size,
        num_kv_heads=flattend_batch_size,
        query_seq_len=query_shape[-2],
        head_size_kv=key_shape[-1],
        head_size=query_shape[-1],
        kv_seq_len=key_shape[-2],
    )
    vanilla_attention = get_wave_kernel(
        shape,
        is_causal=is_causal,
        is_v_transposed=is_v_transposed,
        sliding_window_size=sliding_window_size,
    )
    vanilla_attention.options.dynamic_symbols_map = {
        tkl.sym.B: flattend_batch_size,
        tkl.sym.M: shape.query_seq_len,
        tkl.sym.N: shape.head_size_kv,
        tkl.sym.K2: shape.kv_seq_len,
    }

    _ = vanilla_attention(
        query.view(flat_q_shape),
        key.view(flat_kv_shape),
        value.view(flat_kv_shape),
        output,
    )
    return output.view(*batch, shape.query_seq_len, shape.head_size_kv)


if __name__ == "__main__":
    query = torch.randn([2, 8, 128, 128], device="cuda:0", dtype=torch.float16)
    key = torch.randn([2, 8, 128, 128], device="cuda:0", dtype=torch.float16)
    value = torch.randn([2, 8, 128, 128], device="cuda:0", dtype=torch.float16)
    wave_sdpa(query, key, value)
