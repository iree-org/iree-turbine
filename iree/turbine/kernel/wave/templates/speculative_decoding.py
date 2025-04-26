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
):

    B = tkl.sym.B
    N = tkl.sym.N
    D = tkl.sym.D
    BLOCK_D = tkl.sym.BLOCK_D
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_N = tkl.sym.BLOCK_N
    LAST_OFFSET = tkl.sym.LAST_OFFSET

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

    o_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N: j, D: k},
        outputs={B: i, N: LAST_OFFSET, D: k},
    )
    u_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: i, N: j},
        outputs={B: i, N: LAST_OFFSET},
    )

    uniform_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: i, D: sympy.Integer(0)},
        outputs={B: i, D: j},
    )

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        q: tkl.Memory[B, N, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, N, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        cur_prob_offset: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32],
        uniform_sample: tkl.Memory[B, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        relu_diff_out: tkl.Memory[B, N, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        u_out: tkl.Memory[B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        last_offset = tkw.read(cur_prob_offset, elements_per_thread=1)
        tkw.set_symbol(LAST_OFFSET, last_offset)

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
        tkw.write(relu_diff, relu_diff_out, mapping=o_mapping)
        tkw.write(coin * sum_relu, u_out, mapping=u_mapping)

    hyperparams = {
        BLOCK_B: 1,
        BLOCK_N: 1,
        BLOCK_D: 64,
        B: batch_size,
        N: num_draft_tokens,
        D: d,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return tree_speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map


def get_speculative_sampling_kernel(
    batch_size: int,
    num_speculative_tokens: int,
    threshold_acc: float,
    threshold_single: float,
):
    CUR_INDEX = sympy.Symbol("CUR_INDEX")
    J = tkl.sym.J
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
            vector_shapes={B: batch_size, J: 0, CUR_INDEX: 0, S: 0, D: 20},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 0)]
    constraints += [tkw.TilingConstraint(CUR_INDEX)]
    constraints += [tkw.TilingConstraint(J)]

    hyperparams = {
        BLOCK_B: 1,
        B: batch_size,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        S: 10,
        BLOCK_S: 1,
        D: 20,
    }
    dynamic_symbols = []
    dynamic_symbols_map = {}

    CUR_PROB_OFFSET = tkl.sym.CUR_PROB_OFFSET
    DRAFT_TOKEN_ID = tkl.sym.DRAFT_TOKEN_ID
    NUM_ACCEPTED_TOKENS = tkl.sym.NUM_ACCEPTED_TOKENS
    INNER_DONE = sympy.Symbol("INNER_DONE")
    OUTER_DONE = sympy.Symbol("OUTER_DONE")

    # =================================================================================
    # IndexMapping
    # =================================================================================
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    read_zero_offset_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: sympy.Integer(0), D: sympy.Integer(0)},
        outputs={S: i, B: j, D: k},
    )

    read_at_cur_index_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_INDEX, D: sympy.Integer(0)},
        outputs={S: i, B: j, D: k},
    )

    read_target_probs_at_cur_prob_offset_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_PROB_OFFSET, D: DRAFT_TOKEN_ID},
        outputs={S: i, B: j, D: k},
    )

    write_zero_offset_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, D: k},
        outputs={S: i, B: sympy.Integer(0), D: sympy.Integer(0)},
    )

    write_to_num_accepted_tokens_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, D: k},
        outputs={S: i, B: NUM_ACCEPTED_TOKENS, D: sympy.Integer(0)},
    )

    # =================================================================================
    # Helper functions.
    # =================================================================================
    def read_with_zero_offset(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_zero_offset_mapping)

    def read_at_cur_index(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_at_cur_index_mapping)

    def read_target_probs_at_cur_prob_offset(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_target_probs_at_cur_prob_offset_mapping)

    def read_target_probs_at_cur_index(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_target_probs_at_cur_index_mapping)

    def write_to_zero_offset(value, memory):
        return tkw.write(value, memory, elements_per_thread=1, mapping=write_zero_offset_mapping)

    def write_to_num_accepted_tokens(value, memory):
        return tkw.write(value, memory, elements_per_thread=1, mapping=write_to_num_accepted_tokens_mapping)

    def broadcast(x):
        return tkw.broadcast(x, target_shape=[S, B, D])

    # Kernel.
    # =================================================================================
    @tkw.wave(constraints)
    def speculative_sampling(
        uniform_samples: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.f32],
        target_probs: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.f32],
        draft_probs: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.f32],
        candidates: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        retrieve_index: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_token: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_sibling: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        # Outputs
        predicts: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        accept_token_num: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        accept_index: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        cur_prob_offset_vec: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        last_accepted_retrieve_idx_vec: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
    ):
        one = tkw.Register[S, B, D, tkl.i32](1)
        one_f32 = tkw.Register[S, B, D, tkl.f32](1.0)
        zero = tkw.Register[S, B, D, tkl.i32](0)
        zero_f32 = tkw.Register[S, B, D, tkl.f32](0.0)

        last_accepted_retrieve_idx = read_with_zero_offset(retrieve_index)
        coin = read_with_zero_offset(uniform_samples)
        write_to_zero_offset(last_accepted_retrieve_idx, accept_index)

        outer_loop_condition = J < num_speculative_tokens
        inner_loop_condition = CUR_INDEX >= 0

        tkw.set_symbol(CUR_INDEX, zero)

        @tkw.iterate(
            J,
            start=one,
            condition=outer_loop_condition,
            init_args=[
                zero,
                zero,
                zero,
                last_accepted_retrieve_idx,
                zero_f32,
                coin,
            ],
        )
        def outer_loop(
            cur_index,
            num_accepted_tokens,
            cur_prob_offset,
            last_accepted_retrieve_idx,
            prob_acc,
            coin,
        ):

            tkw.set_symbol(CUR_INDEX, cur_index)
            cur_index = read_at_cur_index(retrieve_next_token)

            @tkw.iterate(
                CUR_INDEX,
                start=cur_index,
                condition=inner_loop_condition,
                init_args=[
                    cur_index,
                    num_accepted_tokens,
                    cur_prob_offset,
                    last_accepted_retrieve_idx,
                    prob_acc,
                    coin,
                ],
            )
            def inner_loop(
                cur_index,
                num_accepted_tokens,
                cur_prob_offset,
                last_accepted_retrieve_idx,
                prob_acc,
                coin,
            ):

                tkw.set_symbol(CUR_INDEX, cur_index)
                draft_index = read_at_cur_index(retrieve_index)
                draft_token_id = read_at_cur_index(candidates)

                tkw.set_symbol(CUR_PROB_OFFSET, cur_prob_offset)
                tkw.set_symbol(DRAFT_TOKEN_ID, draft_token_id)
                target_prob_single = read_target_probs_at_cur_prob_offset(target_probs)
                prob_acc += target_prob_single

                zero_f32_reg = tkw.Register[S, B, D, tkl.f32](0.0)
                threshold_acc_reg = tkw.Register[S, B, D, tkl.f32](threshold_acc)
                threshold_single_reg = tkw.Register[S, B, D, tkl.f32](threshold_single)
              # coin_threshold = prob_acc / threshold_acc_reg
              # condition1 = coin <= coin_threshold
                condition2 = target_prob_single >= threshold_single_reg

                new_cur_index = tkw.select(condition2 | condition2, cur_index, read_at_cur_index(retrieve_next_sibling))
                new_num_accepted_tokens = tkw.select(condition2 | condition2, num_accepted_tokens + one, num_accepted_tokens)
                new_cur_prob_offset = tkw.select(condition2 | condition2, cur_index, cur_prob_offset)
                new_last_accepted_retrieve_idx = tkw.select(condition2 | condition2, draft_index, last_accepted_retrieve_idx)
                new_prob_acc = tkw.select(condition2 | condition2, zero_f32_reg, prob_acc)
                new_coin = tkw.select(condition2 | condition2, read_at_cur_index(uniform_samples), coin)

                tkw.set_symbol(CUR_INDEX, new_cur_index)
                tkw.set_symbol(NUM_ACCEPTED_TOKENS, new_num_accepted_tokens)

              # @tkw.conditional(condition2 | condition2)
              # def then():
              #     tkw.set_symbol(CUR_PROB_OFFSET, cur_index)

              #     write_to_num_accepted_tokens(draft_index, accept_index)
              #  #  true = tkw.Register[S, B, D, tkl.i32](1)
              #  #  tkw.set_symbol(INNER_DONE, true)

                return (
                    new_cur_index,
                    new_num_accepted_tokens,
                    new_cur_prob_offset,
                    new_last_accepted_retrieve_idx,
                    new_prob_acc,
                    new_coin,
                )

            (
                cur_index,
                num_accepted_tokens,
                cur_prob_offset,
                last_accepted_retrieve_idx,
                prob_acc,
                coin,
            ) = inner_loop

            cur_index = broadcast(cur_index) + one
            num_accepted_tokens = broadcast(num_accepted_tokens) + one
            last_accepted_retrieve_idx = broadcast(last_accepted_retrieve_idx) + one
            cur_prob_offset = broadcast(cur_prob_offset) + one
            prob_acc = broadcast(prob_acc) + one_f32
            coin = broadcast(coin) + one_f32
            tkw.set_symbol(J, cur_index)

            return (
                cur_index,
                num_accepted_tokens,
                cur_prob_offset,
                last_accepted_retrieve_idx,
                prob_acc,
                coin,
            )

        (
            cur_index,
            num_accepted_tokens,
            cur_prob_offset,
            last_accepted_retrieve_idx,
            prob_acc,
            coin,
        ) = outer_loop

        tkw.write(
            last_accepted_retrieve_idx,
            last_accepted_retrieve_idx_vec,
            elements_per_thread=1,
        )
        tkw.write(cur_prob_offset, cur_prob_offset_vec, elements_per_thread=1)
        tkw.write(num_accepted_tokens, accept_token_num, elements_per_thread=1)
        tkw.write(cur_index, predicts, elements_per_thread=1)

    return speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
