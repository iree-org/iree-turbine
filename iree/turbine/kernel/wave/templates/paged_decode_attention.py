# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
import sympy
from enum import Enum
from collections import namedtuple
import math

paged_decode_attention_shape = namedtuple(
    "paged_decode_attention_shape",
    [
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "head_size_kv",
        "block_size",
        "num_seqs",
        "kv_lens",
    ],
)


def get_paged_decode_attention_kernels(
    shape: paged_decode_attention_shape,
    mfma_variant: tuple[MMAType, MMAType],
    num_kv_splits: int,
    k_shape: tuple[int],
    v_shape: tuple[int],
    block_table_shape: tuple[int],
):
    # Input sizes
    S = tkl.sym.S  # Num seqs
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    SEQ_LEN = tkl.sym.SEQ_LEN
    SPLIT_OFF = tkl.sym.SPLIT_OFF
    SPLIT_LEN = tkl.sym.SPLIT_LEN
    SPLITS_ACTIVE = tkl.sym.SPLITS_ACTIVE
    U = tkl.sym.U  # Num splits
    BH = tkl.sym.BH
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BH = tkl.sym.BLOCK_BH
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_U = tkl.sym.BLOCK_U
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    class Phase(Enum):
        PHASE_0 = (0,)
        PHASE_1 = (1,)

    THREADS_PER_WAVE = 64
    PHASE_1_BLOCK_B = 64
    PHASE_1_ELEMS_PER_THREAD = PHASE_1_BLOCK_B // THREADS_PER_WAVE
    PHASE_1_BLOCK_N = 1
    HEAD_BLOCK_SIZE = 16
    head_ratio = shape.num_query_heads // shape.num_kv_heads
    B_WAVES = 1

    def phase_0_constraints():
        # K1, K2 are reduction dimensions that are fixed (not distributed) so
        # they are not part of the constraints.

        constraints: list[tkw.Constraint] = []
        # U represents the number of splits of the key-value sequence.
        # U is parallelizable and is distributed across workgroups.
        constraints += [tkw.WorkgroupConstraint(U, BLOCK_U, 2)]
        constraints += [
            tkw.TilingConstraint(
                K2, BLOCK_K2, iters=sympy.ceiling(SPLIT_LEN / BLOCK_K2), start=SPLIT_OFF
            )
        ]

        # BH is the kv-head index and is distributed across workgroups.
        # B is the query index and is distributed like BH but with a different
        # workgroup and wave tile size.

        # For GQA, the number of query heads >> number of kv heads. So we launch
        # workgroups where each workgroup processes HEAD_BLOCK_SIZE query heads
        # as this allows us to use MMA for the attention computation. While
        # each workgroup processes HEAD_BLOCK_SIZE query heads, it only processes
        # one kv head. So we need to specify an apply_func to determine the
        # appropriate kv head index.

        wg_func_2 = lambda wg: wg // math.ceil(head_ratio / HEAD_BLOCK_SIZE)
        count = shape.num_query_heads // min(HEAD_BLOCK_SIZE, head_ratio)
        constraints += [
            tkw.WorkgroupConstraint(BH, BLOCK_BH, 1, apply_fn=wg_func_2, iters=count)
        ]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 1, primary=False)]
        constraints += [tkw.WaveConstraint(B, BLOCK_B / B_WAVES)]

        constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 0)]

        vector_shapes = {BH: 0, S: 0, U: 1}
        waves_per_block = (1, B_WAVES, 1)
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=THREADS_PER_WAVE,
                waves_per_block=waves_per_block,
                mma_type=mfma_variant[1],
                vector_shapes=vector_shapes,
            )
        ]
        return constraints

    def phase_1_constraints() -> list[tkw.Constraint]:
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
        constraints += [tkw.WaveConstraint(B, BLOCK_B)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N)]
        constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 2)]
        constraints += [tkw.TilingConstraint(U, BLOCK_U, iters=SPLITS_ACTIVE)]
        vector_shapes = {
            S: 0,
            B: BLOCK_B,
            N: BLOCK_N,
            U: 1,
        }
        waves_per_block = (1, 1, 1)
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=THREADS_PER_WAVE,
                waves_per_block=waves_per_block,
                mma_type=mfma_variant,
                vector_shapes=vector_shapes,
            )
        ]
        return constraints

    def get_constraints(phase: Phase) -> list[tkw.Constraint]:
        if phase == Phase.PHASE_0:
            return phase_0_constraints()
        else:
            return phase_1_constraints()

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    d0 = tkw.IndexMapping.dynamic_val(0)

    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, N: k},
        outputs={S: i, B: j, N: k},
    )

    K2_dim = k_shape[1]
    # Returns the key for the given token index.
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={S: d0 // K2_dim, BH: j, K2: d0 % K2_dim, K1: l},
        outputs={S: i, BH: j, K2: k, K1: l},
        dynamic_val_mappings={K2: k},
    )

    # Returns the value for the given token index.
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={S: d0 // K2_dim, BH: j, N: k, K2: d0 % K2_dim},
        outputs={S: i, BH: j, N: k, K2: l},
        dynamic_val_mappings={K2: l},
    )

    # Returns token indices into the k-v cache for the given sequence (d0).
    block_table_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={S: d0, K2: i},
        outputs={K2: i},
        dynamic_val_mappings={S: 0},
    )

    k_layout = tkl.MemoryLayout(shape=k_shape)
    v_layout = tkl.MemoryLayout(shape=v_shape)
    block_table_layout = tkl.MemoryLayout(shape=block_table_shape)

    # The kv-cache layout here is (SEQ, HEADS, HEAD_DIM).
    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[S, K2, BH, K1, ADDRESS_SPACE, tkl.f16, k_layout],
        v: tkl.Memory[S, K2, BH, N, ADDRESS_SPACE, tkl.f16, v_layout],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        block_table: tkl.Memory[
            S, K2, GLOBAL_ADDRESS_SPACE, tkl.i32, block_table_layout
        ],
        output: tkl.Memory[U, S, N, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        # =========================================================================
        # Query has shape [NUM_SEQS, NUM_HEADS, HEAD_DIM]
        # Key has shape [NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
        # Value has shape [NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE]
        #                 (TODO: This is a transposed version of the original)
        # Sequence lengths has shape [NUM_SEQS]
        # Request indices has shape [NUM_SEQS]
        # Block table has shape [NUM_SEQS, MAX_KV_SEQ_LEN]
        # Output has shape [NUM_KV_SPLITS, NUM_SEQS, NUM_HEADS, HEAD_DIM]
        # =========================================================================

        init_max = tkl.Register[S, B, tkl.f32](-1e6)
        init_sum = tkl.Register[S, B, tkl.f32](0.0)
        new_acc = tkl.Register[S, N, B, tkl.f32](0.0)

        zero = tkl.Register[B, K2, tkl.f32](0.0)
        neg_infinity = tkl.Register[B, K2, tkl.f32](-1e6)

        # The request index is used to load the appropriate entries from the block table.
        req_index = tkw.read(request_indices)
        # The sequence length is used to control the bounds of the loop over K2.
        seq_length = tkw.read(sequence_lengths)
        tkw.set_symbol(SEQ_LEN, seq_length)

        seq_length_per_split = tkw.apply_expr(
            seq_length, lambda x: sympy.ceiling(x / U)
        )
        seq_length_per_split = tkw.cast(seq_length_per_split, tkl.i32)
        split_offset = tkw.self_index(U, tkl.i32)
        split_offset = tkw.broadcast(split_offset, target_shape=[S, U])
        split_offset = split_offset * seq_length_per_split
        tkw.set_symbol(SPLIT_OFF, split_offset)

        seq_length_per_split = tkw.broadcast(seq_length_per_split, target_shape=[S, U])
        seq_length = tkw.broadcast(seq_length, target_shape=[S, U])
        seq_length_per_split = tkw.apply_expr(
            [seq_length_per_split, seq_length, split_offset],
            lambda x, y, z: sympy.Min(x, sympy.Max(y - z, 0)),
        )
        tkw.set_symbol(SPLIT_LEN, seq_length_per_split)

        @tkw.iterate(K2, init_args=[init_max, init_sum, new_acc])
        def loop(
            partial_max: tkl.Register[S, B, tkl.f32],
            partial_sum: tkl.Register[S, B, tkl.f32],
            acc: tkl.Register[S, N, B, tkl.f32],
        ):
            q_reg = tkw.read(q)
            block_indices_v = tkw.read(
                block_table,
                mapping=block_table_mapping,
                mapping_dynamic_vals=(req_index,),
            )
            block_indices_k = tkw.read(
                block_table,
                mapping=block_table_mapping,
                mapping_dynamic_vals=(req_index,),
            )
            k_reg = tkw.read(
                k,
                mapping=k_mapping,
                mapping_dynamic_vals=(block_indices_k,),
            )
            imm_reg = tkl.Register[S, K2, B, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[S, B, K2])
            k2_index = tkw.self_index(K2, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < (SPLIT_OFF + SPLIT_LEN))
            mask = tkw.broadcast(mask, target_shape=[B, K2])
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, zero, neg_infinity)
            x_j = x_j + bias
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v,
                mapping=v_mapping,
                mapping_dynamic_vals=(block_indices_v,),
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = loop

        @tkw.conditional(SPLIT_LEN > 0)
        def then():
            reciprocal_sum = tkw.reciprocal(res_sum)
            res = res_mm * reciprocal_sum
            res_max_log_sum = res_max + tkw.log2(res_sum)

            tkw.write(res_max_log_sum, output_max)
            tkw.write(res, output)

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, S, N, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        seq_length = tkw.read(sequence_lengths, elements_per_thread=1)
        splits_active = tkw.apply_expr(seq_length, lambda x: sympy.Min(x, U))
        tkw.set_symbol(SPLITS_ACTIVE, splits_active)

        c_reg = tkl.Register[S, B, N, tkl.f32](0.0)
        init_sum = tkl.Register[S, B, tkl.f32](0.0)
        init_max = tkl.Register[S, B, tkl.f32](-1e6)

        @tkw.iterate(U, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[S, B, tkl.f32],
            partial_sum: tkl.Register[S, B, tkl.f32],
            acc: tkl.Register[S, B, N, tkl.f32],
        ):
            x_j = tkw.read(logits, elements_per_thread=PHASE_1_ELEMS_PER_THREAD)
            xm_j = tkw.read(logits_max, elements_per_thread=PHASE_1_ELEMS_PER_THREAD)
            m_j = tkw.maximum(xm_j, partial_max)
            old_scale = tkw.exp2(partial_max - m_j)
            new_scale = tkw.exp2(xm_j - m_j)
            d_j = partial_sum * old_scale + new_scale
            new_acc = acc * old_scale
            term = new_scale * x_j
            new_acc = new_acc + term

            return m_j, d_j, new_acc

        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        res_f16 = tkw.cast(res, tkl.f16)
        tkw.write(
            res_f16,
            output,
            mapping=mapping,
            elements_per_thread=PHASE_1_ELEMS_PER_THREAD,
        )

    symbols_0 = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_BH: 1,
        BLOCK_B: HEAD_BLOCK_SIZE,
        BLOCK_S: 1,
        BLOCK_U: 1,
        BLOCK_K2: 16,
        B: shape.num_query_heads,
        M: 1,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_lens,
        BH: shape.num_kv_heads,
        S: shape.num_seqs,
        U: num_kv_splits,
    }
    symbols_1 = dict(symbols_0)
    symbols_1[BLOCK_B] = PHASE_1_BLOCK_B
    symbols_1[BLOCK_N] = PHASE_1_BLOCK_N

    return (
        phase_0,
        phase_1,
        symbols_0,
        symbols_1,
    )
