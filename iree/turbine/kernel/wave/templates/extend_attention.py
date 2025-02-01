# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
import sympy
from .attention_common import *
import math


def get_extend_attention_kernels(
    shape: AttentionShape,
    mfma_variant: MMAType,
    k_shape: tuple[int],
    v_shape: tuple[int],
    block_table_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
):
    # Input sizes
    S = tkl.sym.S
    # Workgroup tile sizes
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = tkl.sym.LOAD_ELEMS_PER_THREAD_QK
    LOAD_ELEMS_PER_THREAD_PV = tkl.sym.LOAD_ELEMS_PER_THREAD_PV
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    # Dynamic symbols
    REQ_IDX = tkl.sym.REQ_IDX
    SEQ_IDX = tkl.sym.SEQ_IDX
    EXT_IDX = tkl.sym.EXT_IDX

    M_WAVES = 4
    N_WAVES = 1
    THREADS_PER_WAVE = 64
    SEQ_TILE_SIZE = 64

    constraints: list[tkw.Constraint] = []

    constraints += [
        tkw.WorkgroupConstraint(
            N_Q, BLOCK_N_Q, 0, iters=math.ceil(shape.max_seq_len / SEQ_TILE_SIZE)
        )
    ]
    constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 2)]
    constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 3)]
    constraints += [tkw.TilingConstraint(N_KV, BLOCK_N_KV)]
    constraints += [tkw.WaveConstraint(N_Q, BLOCK_N_Q / M_WAVES)]
    constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV / N_WAVES)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    vector_shapes = {S: 0, H: 0, N_Q: Mvec, D_KV: Nvec}
    waves_per_block = (M_WAVES, N_WAVES, 1)
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=THREADS_PER_WAVE,
            waves_per_block=waves_per_block,
            mma_type=mfma_variant[1],
            vector_shapes=vector_shapes,
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    o_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, D_KV: j, N_Q: k},
        outputs={N_Q: k + EXT_IDX, H: i, D_KV: j},
    )

    q_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={N_Q: i + EXT_IDX, H: j, D_Q: k},
        outputs={N_Q: i, H: j, D_Q: k},
    )

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    # Returns the key for the given token index.
    k_mapping_func = lambda x: tkw.IndexMapping(
        num_iterators=3,
        inputs={N_KV: i + x, H: j // head_ratio, D_Q: k},
        outputs={N_KV: i, H: j, D_Q: k},
    )
    k_mapping = k_mapping_func(REQ_IDX)
    k_mapping_ext = k_mapping_func(EXT_IDX)

    # Returns the value for the given token index.
    v_mapping_func = lambda x: tkw.IndexMapping(
        num_iterators=3,
        inputs={N_KV: i + x, H: j // head_ratio, D_KV: k},
        outputs={N_KV: i, H: j, D_KV: k},
    )
    v_mapping = v_mapping_func(REQ_IDX)
    v_mapping_ext = v_mapping_func(EXT_IDX)

    # Returns token indices into the k-v cache for the given sequence (d0).
    # TODO: Verify the stride here.
    block_table_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i + REQ_IDX * block_table_shape[0], N_KV: j},
        outputs={S: i, N_KV: j},
    )

    k_layout = tkl.MemoryLayout(shape=k_shape)
    v_layout = tkl.MemoryLayout(shape=v_shape)
    block_table_layout = tkl.MemoryLayout(shape=block_table_shape)
    k_cache_layout = tkl.MemoryLayout(shape=k_cache_shape)
    v_cache_layout = tkl.MemoryLayout(shape=v_cache_shape)
    o_layout = tkl.MemoryLayout(shape=o_shape)

    @tkw.wave(constraints)
    def extend(
        q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[N_KV, H, D_Q, ADDRESS_SPACE, tkl.f16, k_layout],
        v: tkl.Memory[H, D_KV, N_KV, ADDRESS_SPACE, tkl.f16, v_layout],
        k_cache: tkl.Memory[
            N_KV, H, D_Q, GLOBAL_ADDRESS_SPACE, tkl.f16, k_cache_layout
        ],
        v_cache: tkl.Memory[
            N_KV, H, D_KV, GLOBAL_ADDRESS_SPACE, tkl.f16, v_cache_layout
        ],
        block_table: tkl.Memory[
            S, N_KV, GLOBAL_ADDRESS_SPACE, tkl.i32, block_table_layout
        ],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        sequence_lengths_extend: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        start_indices_extend: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        output: tkl.Memory[N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, tkl.f32, o_layout],
    ):

        req_index = tkw.read(request_indices, elements_per_thread=1)
        tkw.set_symbol(REQ_IDX, req_index)
        start_loc_extend = tkw.read(start_indices_extend, elements_per_thread=1)
        tkw.set_symbol(EXT_IDX, start_loc_extend)

        seq_len = tkw.read(sequence_lengths, elements_per_thread=1)
        seq_len_extend = tkw.read(sequence_lengths_extend, elements_per_thread=1)
        seq_len_prefix = seq_len - seq_len_extend

        tkw.set_symbol(N_KV, seq_len_prefix)

        init_max = tkl.Register[H, N_Q, tkl.f32](-1e6)
        init_sum = tkl.Register[H, N_Q, tkl.f32](0.0)
        new_acc = tkl.Register[H, D_KV, N_Q, tkl.f32](0.0)

        @tkw.reduction(N_KV, init_args=[init_max, init_sum, new_acc])
        def loop(
            partial_max: tkl.Register[H, N_Q, tkl.f32],
            partial_sum: tkl.Register[H, N_Q, tkl.f32],
            acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
        ):
            block_indices = tkw.read(
                block_table,
                elements_per_thread=1,
                mapping=block_table_mapping,
            )
            tkw.set_symbol(SEQ_IDX, block_indices)
            q_reg = tkw.read(
                q, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK, mapping=q_mapping
            )
            k_reg = tkw.read(
                k_cache,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=k_mapping,
            )
            imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v_cache,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=v_mapping,
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = loop
        # TODO: For a causal mask, we can define a new symbol N_KV_CAUSAL
        # and set it here for the reduction below it. The count of the
        # associated TilingConstraint below must be adjusted to be
        # min(seq_len_extend, WG_ID(N_Q) * BLOCK_N_Q).
        tkw.set_symbol(N_KV, seq_len_extend)

        # This loop is identical to prefill.
        @tkw.reduction(N_KV, init_args=[res_max, res_sum, res_mm])
        def second_loop(
            partial_max: tkl.Register[H, N_Q, tkl.f32],
            partial_sum: tkl.Register[H, N_Q, tkl.f32],
            acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
        ):
            q_reg = tkw.read(
                q, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK, mapping=q_mapping
            )
            k_reg = tkw.read(
                k,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=k_mapping_ext,
            )
            imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
            # TODO: Add causal mask here.
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=v_mapping_ext,
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = second_loop
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum

        tkw.write(
            res, output, mapping=o_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    symbols = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK: get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV: get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_H: 1,
        BLOCK_S: 1,
        BLOCK_D_KV: SEQ_TILE_SIZE,
        BLOCK_N_Q: SEQ_TILE_SIZE,
        BLOCK_N_KV: SEQ_TILE_SIZE,
        H: shape.num_query_heads,
        D_Q: shape.head_size,
        D_KV: shape.head_size_kv,
        S: shape.num_seqs,
        N_Q: o_shape[0],
    }

    return (
        extend,
        symbols,
    )
