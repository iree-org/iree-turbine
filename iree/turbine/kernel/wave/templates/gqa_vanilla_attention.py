# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
)
from .attention_common import *
import math
import torch
from typing import Optional


def get_gqa_bshd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    input_dtype: Optional[torch.dtype] = torch.float16,
    output_dtype: Optional[torch.dtype] = torch.float32,
    is_causal: Optional[bool] = False,
    layer_scaling: Optional[float] = None,
    sliding_window_size: Optional[int] = -1,
):

    if sliding_window_size > 0 and not is_causal:
        raise NotImplementedError(
            "Sliding window is only supported for causal attention."
        )

    # Determine dtype of operands.
    wave_input_dtype = torch_dtype_to_wave(input_dtype)
    wave_output_dtype = torch_dtype_to_wave(output_dtype)

    LOG2E = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E

    B = tkl.sym.B
    BLOCK_B = tkl.sym.BLOCK_B
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(N_Q, BLOCK_N_Q, 0)]
    constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.WorkgroupConstraint(H_KV, BLOCK_H, 3, primary=False)]
    constraints += [tkw.TilingConstraint(N_KV, BLOCK_N_KV)]
    constraints += [tkw.WaveConstraint(N_Q, BLOCK_N_Q / 4)]
    constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, H: 0, H_KV: 0, N_Q: Mvec, D_KV: Nvec},
        )
    ]

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, D_KV: k, N_Q: l},
        outputs={B: i, N_Q: l, H: j, D_KV: k},
    )
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N_Q: k, D_Q: l},
        outputs={B: i, H: j, N_Q: k, D_Q: l},
    )
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, N_KV: k, D_Q: l},
        outputs={B: i, H_KV: j, N_KV: k, D_Q: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j // head_ratio, D_KV: k, N_KV: l},
        outputs={B: i, H_KV: j, D_KV: k, N_KV: l},
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype],
        k: tkl.Memory[B, N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype],
        v: tkl.Memory[B, N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype],
        c: tkl.Memory[B, N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype],
    ):

        qkv_scaling = tkl.Register[B, H, N_Q, D_Q, tkl.f16](layer_scaling)
        c_reg = tkl.Register[B, H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[B, H, N_Q, tkl.f32](-1e6)
        sliding_window = tkl.Register[N_Q, N_KV, tkl.i32](sliding_window_size)
        ZEROF = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
        MIN_INF = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)

        @tkw.reduction(N_KV, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, N_Q, tkl.f32],
            partial_sum: tkl.Register[B, H, N_Q, tkl.f32],
            acc: tkl.Register[B, H, D_KV, N_Q, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, N_KV, N_Q, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k, mapping=k_mapping)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, N_Q, N_KV])
            k2_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            if is_causal:
                m_index = tkw.self_index(N_Q, tkl.i32)
                m_index = tkw.broadcast(m_index, target_shape=[N_Q, N_KV])
                mask = (m_index >= k2_index) & mask
                if sliding_window_size > 0:
                    mask = (m_index - k2_index <= sliding_window) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_N_Q: 128,
        BLOCK_D_KV: 64,
        BLOCK_N_KV: 64,
        B: shape.num_seqs,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        N_Q: shape.query_seq_len,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
        N_KV: shape.kv_seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
