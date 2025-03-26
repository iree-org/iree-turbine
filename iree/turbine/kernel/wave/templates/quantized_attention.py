# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from .attention_common import AttentionShape
import math
from iree.turbine.kernel.wave.utils.general_utils import torch_dtype_to_wave


def get_brevitas_pertensor_fp8_attention_kernel(
    shape: AttentionShape,
    mfma_variant: MMAType,
    logit_dtype: torch.dtype = torch.float16,
    f8_dtpye: torch.dtype = torch.float8_e4m3fnuz,
    dynamic_dims: bool = False,
    is_causal: bool = False,
    q_scale=1.0,
    k_scale=1.0,
    v_scale=1.0,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant[0] == MMAType.F32_16x16x32_F8:
        Mvec = 16
        Nvec = 16
    if mfma_variant[0] == MMAType.F32_32x32x16_F8:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    # Value tensor mapping to transpose for efficient computation if the input is
    # not already transposed.
    v_mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, K2: k}, outputs={B: i, N: j, K2: k}
    )

    # Setting up input DTYPE
    LOGIT_DTYPE = torch_dtype_to_wave(logit_dtype)

    # Setting up FP8 scaling
    LOG2E = 1.44269504089
    DK_SQRT = math.sqrt(1.0 / shape.head_size)
    F8_DTYPE = torch_dtype_to_wave(f8_dtpye)
    F8_MAX = torch.finfo(f8_dtpye).max

    # maximum expected value from attention softmax
    ATTENTION_SOFTMAX_MAX = 1.0

    # FP8 offset
    # If we need to truncate to fp8 post softmax we apply a scaling to use the
    # full fp8 range. We can do this with a offset as post `exp2` this equates
    # to multiplying by a static value. We are able to do this as `max` and
    # `sum` are scaled by the same value so the end result is the same.
    FP8_OFFSET_VAL = ATTENTION_SOFTMAX_MAX / F8_MAX

    # Dequant Tensor Scaling
    DEQUANT_QK = q_scale * k_scale

    # Clamp input to dstTy(usually `fp8`) MAX value to prevent NaNs.
    # We do not clamp for `-MAX` because this function meant to only be
    # used by attention's exp2 who's value is always > 0.
    def low_precision_clamp(source_reg, upper_bound):
        clamped = tkw.minimum(source_reg, upper_bound)
        return tkw.cast(clamped, F8_DTYPE)

    def base_attention_core(q, k, v, c):
        qk_scaling = tkl.Register[B, M, K2, tkl.f32](DK_SQRT * LOG2E * DEQUANT_QK)
        v_dequant = tkl.Register[B, N, M, tkl.f32](v_scale)
        fp8_offset = tkl.Register[B, M, K2, tkl.f32](FP8_OFFSET_VAL)
        fp8_max = tkl.Register[B, M, K2, tkl.f32](F8_MAX)
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            if logit_dtype != F8_DTYPE:
                q_reg = tkw.cast(q_reg, F8_DTYPE)
                k_reg = tkw.cast(k_reg, F8_DTYPE)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            k2_index = tkw.self_index(K2, tkl.i64)
            mask = tkw.apply_expr(k2_index, lambda x: x < K2)
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            if is_causal:
                # Indices i and j broadcasted along K2 with a twist:
                # here we use *static* information that is *implicitly* encoded
                # in the *transformation*: under the distribution constraints
                # specified we know that the shape [M] will eventually resolve
                # to [1] and can thus be "cast + broadcast" to [K2].
                m_index = tkw.self_index(M, tkl.i64)
                m_index = tkw.broadcast(m_index, target_shape=[M, K2])
                mask = (m_index >= k2_index) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            x_j *= qk_scaling
            x_j += fp8_offset
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f8 = low_precision_clamp(e_delta, fp8_max)
            v_reg = tkw.read(
                v, elements_per_thread=LOAD_ELEMS_PER_THREAD_PV, mapping=v_mapping
            )
            if logit_dtype != F8_DTYPE:
                v_reg = tkw.cast(v_reg, F8_DTYPE)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f8, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum * v_dequant
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, LOGIT_DTYPE],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, LOGIT_DTYPE],
        v: tkl.Memory[B, K2, N, ADDRESS_SPACE, LOGIT_DTYPE],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK: get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV: get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape.num_query_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
