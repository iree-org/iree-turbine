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
from iree.turbine.kernel.wave.templates.quantized_attention import (
    get_brevitas_pertensor_fp8_attention_kernel,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape


ATTENTION_SUPPORTED_DTYPE = {torch.float16}


@functools.lru_cache
def get_wave_kernel(
    shape: AttentionShape,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    logit_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    is_causal: bool,
):
    assert shape.num_query_heads % shape.num_kv_heads == 0

    mfma_variant = (MMAType.F32_16x16x32_F8, MMAType.F32_16x16x32_K4_F8)
    (
        quantized_attention,
        hyperparams,
        _,
        _,
    ) = get_brevitas_pertensor_fp8_attention_kernel(
        shape,
        mfma_variant,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        is_causal=is_causal,
        logit_dtype=logit_dtype,
        f8_dtype=quant_dtype,
    )
    hyperparams.update(get_default_scheduling_params())
    del hyperparams[tkl.sym.B]
    del hyperparams[tkl.sym.N_Q]
    del hyperparams[tkl.sym.N_KV]
    dynamic_symbols = [tkl.sym.B, tkl.sym.N_Q, tkl.sym.N_KV]

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map={},
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    quantized_attention = wave_compile(options, quantized_attention)
    return quantized_attention


def wave_sdpa_fp8(
    query,
    key,
    value,
    q_scale,
    k_scale,
    v_scale,
    quant_dtype: torch.dtype = torch.float8_e4m3fnuz,
    is_causal=False,
):
    """
    SDPA FP8 op that Handles any dimension of batch as long as SxD is fastest dim.
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
    quantized_attention = get_wave_kernel(
        shape,
        q_scale,
        k_scale,
        v_scale,
        logit_dtype=query.dtype,
        quant_dtype=quant_dtype,
        is_causal=is_causal,
    )
    quantized_attention.options.dynamic_symbols_map = {
        tkl.sym.B: flattend_batch_size,
        tkl.sym.N_Q: query_shape[-2],
        tkl.sym.N_KV: key_shape[-2],
    }
    _ = quantized_attention(
        query.view(flat_q_shape),
        key.view(flat_kv_shape),
        value.view(flat_kv_shape),
        output,
    )
    return output.view(*batch, query_shape[-2], key_shape[-1])


if __name__ == "__main__":
    query = torch.randn([2, 8, 128, 128], device="cuda:0")
    key = torch.randn([2, 8, 128, 128], device="cuda:0")
    value = torch.randn([2, 8, 128, 128], device="cuda:0")
    q_scale = 0.02578124962747097
    k_scale = 0.02363281324505806
    v_scale = 0.010286458767950535
    wave_sdpa_fp8(query, key, value, q_scale, k_scale, v_scale)
