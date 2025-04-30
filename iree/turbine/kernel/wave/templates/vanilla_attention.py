# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from .attention_common import AttentionShape
from dataclasses import dataclass
import math


def get_vanilla_attention_kernel(
    shape: AttentionShape,
    mfma_variant: MMAType,
    dynamic_dims: bool,
    is_causal: bool = False,
    is_v_transposed: bool = False,
    sliding_window_size: int = -1,
    scale: float = None,
):

    if sliding_window_size > 0 and not is_causal:
        raise NotImplementedError(
            "Sliding window is only supported for causal attention."
        )

    scale = scale or (1.0 / math.sqrt(shape.head_size))
    scale *= math.log2(math.e)

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

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

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

    def base_attention_core(q, k, v, c):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        sliding_window = tkl.Register[M, K2, tkl.i64](sliding_window_size)
        qk_scaling = tkl.Register[B, M, K2, tkl.f32](scale)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            k_reg = tkw.read(k)
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
                if sliding_window_size > 0:
                    mask = (m_index - k2_index <= sliding_window) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            x_j *= qk_scaling
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            if is_v_transposed:
                v_reg = tkw.read(v)
            else:
                v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    @tkw.wave(constraints)
    def base_attention_transposed_v(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
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

    if is_v_transposed:
        base_attention = base_attention_transposed_v

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map


def get_bshd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    dynamic_dims: bool,
    is_causal: bool = False,
    is_custom_mask: bool = False,
):
    # Input sizes
    # BxS_QxHxD x BxS_KVxH_KVxD
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    H = tkl.sym.H
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_H = tkl.sym.BLOCK_H
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

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
            vector_shapes={B: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N: k, M: l},
        outputs={B: i, M: l, H: j, N: k},
    )
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, M: k, K1: l},
        outputs={B: i, H: j, M: k, K1: l},
    )
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, K2: k, K1: l},
        outputs={B: i, H: j, K2: k, K1: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N: k, K2: l},
        outputs={B: i, H: j, N: k, K2: l},
    )

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    def base_attention_core(q, k, v, c):
        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * log2e)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e6)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k, mapping=k_mapping)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < K2)
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            if is_causal:
                m_index = tkw.self_index(M, tkl.i32)
                m_index = tkw.broadcast(m_index, target_shape=[M, K2])
                mask = (m_index >= k2_index) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
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

    def base_attention_core_custom_mask(q, k, v, custom_mask, c):
        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * log2e)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e5)
        ZEROF = tkl.Register[B, M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[B, M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k, mapping=k_mapping)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x >= K2)
            mask = tkw.broadcast(mask, target_shape=[B, M, K2])
            mask = tkw.cast(mask, tkw.i1)

            if is_custom_mask:
                custom_mask_tensor = tkw.read(custom_mask)
                custom_mask_tensor = tkw.broadcast(
                    custom_mask_tensor,
                    target_shape=[B, M, K2],
                )
                custom_mask_tensor = tkw.cast(custom_mask_tensor, tkw.i1)
                mask = mask | custom_mask_tensor

            bias = tkw.select(mask, MIN_INF, ZEROF)

            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)

            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)

            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)

            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)

        # we are handling nan case here when all the tokens are masked and res_sum is 0.0
        is_nan = res_sum == init_sum
        is_nan = tkw.cast(is_nan, tkw.i1)
        upd_reciprocal_sum = tkw.select(is_nan, init_sum, reciprocal_sum)

        res = res_mm * upd_reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, H, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, H, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    @tkw.wave(constraints)
    def base_attention_custom_mask(
        q: tkl.Memory[B, M, H, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, H, N, ADDRESS_SPACE, tkl.f16],
        custom_mask: tkl.Memory[B, M, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core_custom_mask(q, k, v, custom_mask, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: 1,
        H: shape.num_query_heads,
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

    if is_custom_mask:
        return (
            base_attention_custom_mask,
            hyperparams,
            dynamic_symbols,
            dynamic_symbols_map,
        )
    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map


def get_bhsd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    dynamic_dims: bool,
    is_causal: bool = False,
    is_custom_mask: bool = False,
):
    # Input sizes
    # BxHxS_QxD x BxH_KVxS_KVxD
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    H = tkl.sym.H
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_H = tkl.sym.BLOCK_H
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

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
            vector_shapes={B: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, M: k, N: l},
        outputs={B: i, H: j, M: k, N: l},
    )
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N: k, K2: l},
        outputs={B: i, H: j, N: k, K2: l},
    )

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    def base_attention_core(q, k, v, c):
        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * log2e)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e6)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < K2)
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            if is_causal:
                m_index = tkw.self_index(M, tkl.i32)
                m_index = tkw.broadcast(m_index, target_shape=[M, K2])
                mask = (m_index >= k2_index) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
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

    def base_attention_core_custom_mask(q, k, v, custom_mask, c):
        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * log2e)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e5)
        ZEROF = tkl.Register[B, M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[B, M, K2, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling
            k_reg = tkw.read(k, mapping=k_mapping)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x >= K2)
            mask = tkw.broadcast(mask, target_shape=[B, M, K2])
            mask = tkw.cast(mask, tkw.i1)

            if is_custom_mask:
                custom_mask_tensor = tkw.read(custom_mask)
                custom_mask_tensor = tkw.broadcast(
                    custom_mask_tensor,
                    target_shape=[B, M, K2],
                )
                custom_mask_tensor = tkw.cast(custom_mask_tensor, tkw.i1)
                mask = mask | custom_mask_tensor

            bias = tkw.select(mask, MIN_INF, ZEROF)

            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)

            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)

            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)

            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)

        # we are handling nan case here when all the tokens are masked and res_sum is 0.0
        if is_custom_mask:
            is_nan = res_sum == init_sum
            is_nan = tkw.cast(is_nan, tkw.i1)
            upd_reciprocal_sum = tkw.select(is_nan, init_sum, reciprocal_sum)

            res = res_mm * upd_reciprocal_sum
        else:
            res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=mapping)

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, H, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, H, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, H, K2, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, H, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core(q, k, v, c)

    @tkw.wave(constraints)
    def base_attention_custom_mask(
        q: tkl.Memory[B, M, H, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2, H, N, ADDRESS_SPACE, tkl.f16],
        custom_mask: tkl.Memory[B, M, GLOBAL_ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        base_attention_core_custom_mask(q, k, v, custom_mask, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape.batch_size,
        H: shape.num_query_heads,
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

    if is_custom_mask:
        return (
            base_attention_custom_mask,
            hyperparams,
            dynamic_symbols,
            dynamic_symbols_map,
        )
    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
