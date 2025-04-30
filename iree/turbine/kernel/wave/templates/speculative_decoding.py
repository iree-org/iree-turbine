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
import sympy


def get_speculative_decoding_kernel(
    batch_size: int,
    num_draft_tokens: int,
    d: int,
    seq_len: int,
):

    B = tkl.sym.B
    N = tkl.sym.N
    D = tkl.sym.D
    S = tkl.sym.S
    BLOCK_D = tkl.sym.BLOCK_D
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_N = tkl.sym.BLOCK_N
    LAST_OFFSET = tkl.sym.LAST_OFFSET
    LAST_IDX = tkl.sym.LAST_IDX

    constraints = [tkw.WorkgroupConstraint(D, BLOCK_D, 0)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1, primary=False)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 0, N: 0, D: 64},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    q_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N: LAST_OFFSET, D: k},
        outputs={B: i, N: j, D: k},
    )

    p_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N: LAST_OFFSET, D: k},
        outputs={B: i, N: j, D: k},
    )

    uniform_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: i, D: sympy.Integer(0)},
        outputs={B: i, D: j},
    )

    o_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: i, N: j},
        outputs={S: LAST_IDX},
    )

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        q: tkl.Memory[B, N, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, N, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        cur_prob_offset: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32],
        uniform_sample: tkl.Memory[B, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        last_accepted_retrive_idx_vec: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32],
        predicts: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        last_offset = tkw.read(cur_prob_offset, elements_per_thread=1)
        tkw.set_symbol(LAST_OFFSET, last_offset)

        last_idx = tkw.read(last_accepted_retrive_idx_vec, elements_per_thread=1)
        tkw.set_symbol(LAST_IDX, last_idx)

        q_reg = tkw.read(q, mapping=q_mapping)
        p_reg = tkw.read(p, mapping=p_mapping)

        # TODO: Add conditioned mask once scalar codegen is landed.
        # mask_cond = num_accepted_tokens != num_speculative_tokens_sub1
        # mask_cond = tkw.broadcast(mask_cond, target_shape=[B, N, D])
        # p_reg = tkw.select(mask_cond, p_reg, zero)

        coin = tkw.read(uniform_sample, mapping=uniform_mapping)
        diff = q_reg - p_reg

        zero = tkl.Register[D, tkl.f32](0.0)
        relu_diff = tkw.maximum(diff, zero)
        sum_relu = tkw.sum(relu_diff, dim=D)
        cdf = tkw.cumsum(relu_diff, dim=D)

        u = tkw.broadcast(coin * sum_relu, target_shape=[B, N, D])
        greater_than_u = cdf > u
        pad_token = tkl.Register[B, N, D, tkl.i32](1e6)
        token_idx = tkl.Register[B, N, D, tkl.i32](THREAD_0)

        # TODO: Set default sampled_id = d - 1, if no valid token can be found
        #       We can implement with `ballot(greater_than_u)` and early exit
        #       /return d-1 if output are all zeros.
        valid_lane_token_idx = tkw.select(greater_than_u, token_idx, pad_token)
        min_valid_token_idx = tkw.min(valid_lane_token_idx, dim=D)
        tkw.write(min_valid_token_idx, predicts, mapping=o_mapping)

    hyperparams = {
        BLOCK_B: 1,
        BLOCK_N: 1,
        BLOCK_D: 64,
        B: batch_size,
        N: num_draft_tokens,
        D: d,
        S: seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return tree_speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map


def get_speculative_sampling_kernel(
    batch_size: int,
    num_speculative_tokens: int,
    threshold_acc: float,
    threshold_single: float,
    num_draft_tokens: int,
    d: int,
):
    CUR_INDEX = sympy.Symbol("CUR_INDEX")
    J = sympy.Symbol("J")
    B = tkl.sym.B
    S = tkl.sym.S
    D = tkl.sym.D
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_S = tkl.sym.BLOCK_S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: num_draft_tokens, J: 0, CUR_INDEX: 0, S: 0, D: d},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 0)]
    constraints += [tkw.TilingConstraint(CUR_INDEX)]
    constraints += [tkw.TilingConstraint(J)]

    hyperparams = {
        BLOCK_B: 1,
        B: num_draft_tokens,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        S: batch_size,
        BLOCK_S: 1,
        D: d,
    }
    dynamic_symbols = []
    dynamic_symbols_map = {}

    CUR_PROB_OFFSET = tkl.sym.CUR_PROB_OFFSET
    DRAFT_TOKEN_ID = tkl.sym.DRAFT_TOKEN_ID
    NUM_ACCEPTED_TOKENS = tkl.sym.NUM_ACCEPTED_TOKENS
    LAST_ACCEPTED_RETRIEVE_IDX = tkl.sym.LAST_ACCEPTED_RETRIEVE_IDX
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping_2d = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: CUR_INDEX},
        outputs={S: i, B: j},
    )

    mapping_3d = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_PROB_OFFSET, D: DRAFT_TOKEN_ID},
        outputs={S: i, B: j, D: k},
    )

    mapping_3d_2 = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_INDEX, D: DRAFT_TOKEN_ID},
        outputs={S: i, B: j, D: k},
    )

    read_zero_offset_2d_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: sympy.Integer(0)},
        outputs={S: i, B: j},
    )

    write_mapping_2d = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: j},
        outputs={S: i, B: NUM_ACCEPTED_TOKENS},
    )

    write_mapping_1d = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: j},
        outputs={B: LAST_ACCEPTED_RETRIEVE_IDX},
    )

    write_mapping_3d = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, D: k},
        outputs={S: i, B: CUR_INDEX, D: DRAFT_TOKEN_ID},
    )

    write_zero_offset_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: j},
        outputs={S: i, B: sympy.Integer(0)},
    )

    def broadcast(x):
        return tkw.broadcast(x, target_shape=[S, B, D])

    def read_2d(x):
        return tkw.read(x, elements_per_thread=1, mapping=mapping_2d)

    def read_3d(x):
        return tkw.read(x, elements_per_thread=1, mapping=mapping_3d)

    def read_3d_2(x):
        return tkw.read(x, elements_per_thread=1, mapping=mapping_3d_2)

    def read_with_zero_offset_2d(memory):
        return tkw.read(
            memory, elements_per_thread=1, mapping=read_zero_offset_2d_mapping
        )

    def write_2d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_2d)

    def write_1d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_1d)

    def write_3d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_3d)

    def write_with_zero_offset(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_zero_offset_mapping)

    accept_token_num_layout = tkl.MemoryLayout(shape=[batch_size, 1, 1])
    accept_index_layout = tkl.MemoryLayout(shape=[batch_size, num_speculative_tokens])
    cur_prob_offset_vec_layout = tkl.MemoryLayout(shape=[batch_size, 1, 1])
    last_accepted_retrieve_idx_vec_layout = tkl.MemoryLayout(shape=[batch_size, 1, 1])
    predict_layout = tkl.MemoryLayout(shape=[batch_size * num_draft_tokens])
    # Kernel.
    # =================================================================================
    @tkw.wave(constraints)
    def speculative_sampling(
        uniform_samples: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.f32],
        target_probs: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.f32],
        draft_probs: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.f32],
        candidates: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_index: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_token: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_sibling: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        # Outputs
        predicts: tkl.Memory[B, ADDRESS_SPACE_0, tkl.i32, predict_layout],
        accept_token_num: tkl.Memory[
            S, B, D, ADDRESS_SPACE_0, tkl.i32, accept_token_num_layout
        ],
        accept_index: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32, accept_index_layout],
        cur_prob_offset_vec: tkl.Memory[
            S, B, D, ADDRESS_SPACE_0, tkl.i32, cur_prob_offset_vec_layout
        ],
        last_accepted_retrieve_idx_vec: tkl.Memory[
            S, B, D, ADDRESS_SPACE_0, tkl.i32, last_accepted_retrieve_idx_vec_layout
        ],
    ):
        one = tkw.Register[S, B, D, tkl.i32](1)
        zero = tkw.Register[S, B, D, tkl.i32](0)
        zero_f32 = tkw.Register[S, B, D, tkl.f32](0.0)

        threshold_acc_reg = tkw.Register[S, B, D, tkl.f32](threshold_acc)
        threshold_single_reg = tkw.Register[S, B, D, tkl.f32](threshold_single)

        outer_loop_condition = (J < num_speculative_tokens) & (
            sympy.Eq(GET_ITER_ARG(6), 0)
        )
        inner_loop_condition = (CUR_INDEX >= 0) & (sympy.Eq(GET_ITER_ARG(6), 0))

        coin = read_with_zero_offset_2d(uniform_samples)
        last_accepted_retrieve_idx = read_with_zero_offset_2d(retrieve_index)
        write_with_zero_offset(last_accepted_retrieve_idx, accept_index)
        last_accepted_retrieve_idx = tkw.broadcast(
            last_accepted_retrieve_idx, target_shape=[S, B, D]
        )

        @tkw.iterate(
            J,
            start=one,
            condition=outer_loop_condition,
            init_args=[
                zero,  # cur_index
                zero,  # num_accepted_tokens
                last_accepted_retrieve_idx,
                zero,  # cur_prob_offset
                zero_f32,  # prob_acc
                coin,
                zero,  # outer_done
            ],
        )
        def outer_loop(
            cur_index,
            num_accepted_tokens,
            last_accepted_retrieve_idx,
            cur_prob_offset,
            prob_acc,
            coin,
            outer_done,
        ):

            tkw.set_symbol(CUR_INDEX, cur_index)
            cur_index = read_2d(retrieve_next_token)
            zero = tkw.Register[S, B, D, tkl.i32](0)

            @tkw.iterate(
                CUR_INDEX,
                start=cur_index,
                condition=inner_loop_condition,
                init_args=[
                    cur_index,
                    num_accepted_tokens,
                    last_accepted_retrieve_idx,
                    cur_prob_offset,
                    prob_acc,
                    coin,
                    zero,
                ],
            )
            def inner_loop(
                cur_index,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                prob_acc,
                coin,
                inner_done,
            ):
                tkw.set_symbol(CUR_INDEX, cur_index)
                draft_index = read_2d(retrieve_index)
                draft_token_id = read_2d(candidates)
                tkw.set_symbol(DRAFT_TOKEN_ID, draft_token_id)
                tkw.set_symbol(CUR_PROB_OFFSET, cur_prob_offset)
                target_prob_single = read_3d(target_probs)

                condition = (coin <= (prob_acc / threshold_acc_reg)) | (
                    target_prob_single >= threshold_single_reg
                )
                not_condition = ~condition

                # Update num_accepted_tokens if the condition is true.
                num_accepted_tokens = tkw.select(
                    condition, num_accepted_tokens + one, num_accepted_tokens
                )
                tkw.set_symbol(NUM_ACCEPTED_TOKENS, num_accepted_tokens)
                tkw.set_symbol(LAST_ACCEPTED_RETRIEVE_IDX, last_accepted_retrieve_idx)

                # Update cur_prob_offset.
                cur_prob_offset = tkw.select(
                    condition,
                    broadcast(cur_index),
                    broadcast(cur_prob_offset),
                )

                # Update coin.
                coin = tkw.select(
                    condition,
                    broadcast(read_2d(uniform_samples)),
                    broadcast(coin),
                )

                @tkw.conditional(condition)
                def then_():
                    write_1d(draft_token_id, predicts)
                    write_2d(draft_index, accept_index)

                @tkw.conditional(not_condition)
                def else_():
                    target_prob = read_3d_2(target_probs)
                    write_3d(target_prob, draft_probs)

                # Update last_accepted_retrieve_idx.
                last_accepted_retrieve_idx = tkw.select(
                    condition,
                    broadcast(draft_index),
                    last_accepted_retrieve_idx,
                )

                # Update prob_acc.
                prob_acc = tkw.select(
                    condition,
                    zero_f32,
                    prob_acc + target_prob_single,
                )

                # Update cur_index.
                cur_index = tkw.select(
                    not_condition,
                    broadcast(read_2d(retrieve_next_sibling)),
                    broadcast(cur_index),
                )

                # Update inner_done.
                inner_done = tkw.select(
                    condition,
                    one,
                    inner_done,
                )

                tkw.set_symbol(CUR_INDEX, cur_index)

                return (
                    cur_index,
                    num_accepted_tokens,
                    last_accepted_retrieve_idx,
                    cur_prob_offset,
                    prob_acc,
                    coin,
                    inner_done,
                )

            (
                cur_index,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                prob_acc,
                coin,
                inner_done,
            ) = inner_loop

            current_j = tkw.self_index(J, tkl.i32)
            next_j = tkw.apply_expr(current_j, lambda x: x + 1)
            tkw.set_symbol(J, next_j)

            outer_done = tkw.select(cur_index < zero, one, outer_done)

            return (
                cur_index,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                prob_acc,
                coin,
                outer_done,
            )

        (
            cur_index,
            num_accepted_tokens,
            last_accepted_retrieve_idx,
            cur_prob_offset,
            prob_acc,
            coin,
            outer_done,
        ) = outer_loop

        tkw.write(
            last_accepted_retrieve_idx,
            last_accepted_retrieve_idx_vec,
            elements_per_thread=1,
        )
        tkw.write(cur_prob_offset, cur_prob_offset_vec, elements_per_thread=1)
        tkw.write(num_accepted_tokens, accept_token_num, elements_per_thread=1)

    return speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
