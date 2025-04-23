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
):
    # CUR_INDEX = tkl.sym.CUR_INDEX
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

    CUR_PROB_OFFSET = tkl.sym.CUR_PROB_OFFSET
    DRAFT_TOKEN_ID = tkl.sym.DRAFT_TOKEN_ID
    LAST_ACCEPTED_RETRIEVE_IDX = tkl.sym.LAST_ACCEPTED_RETRIEVE_IDX
    INNER_DONE = sympy.Symbol("INNER_DONE")
    OUTER_DONE = sympy.Symbol("OUTER_DONE")

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    read_zero_offset_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: sympy.Integer(0)},
        outputs={S: i, B: j},
    )

    read_2d_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: i, B: CUR_INDEX},
        outputs={S: i, B: j},
    )

    read_3d_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_PROB_OFFSET, D: DRAFT_TOKEN_ID},
        outputs={S: i, B: j, D: k},
    )

    read_3d_mapping_2 = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: CUR_INDEX, D: DRAFT_TOKEN_ID},
        outputs={S: i, B: j, D: k},
    )

    write_1d_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={S: i, B: DRAFT_TOKEN_ID},
        outputs={S: i, B: j},
    )

    write_3d_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, D: k},
        outputs={S: i, B: CUR_INDEX, D: DRAFT_TOKEN_ID},
    )

    # =================================================================================
    # Helper functions.
    # =================================================================================
    def read_with_zero_offset(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_zero_offset_mapping)

    def read_with_2d_offset(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_2d_mapping)

    def read_with_3d_offset(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_3d_mapping)

    def read_with_3d_offset_2(memory):
        return tkw.read(memory, elements_per_thread=1, mapping=read_3d_mapping_2)

    def write_with_3d_offset(memory, value):
        return tkw.write(value, memory, elements_per_thread=1, mapping=write_3d_mapping)

    # =================================================================================
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
        predicts: tkl.Memory[S, B, D, ADDRESS_SPACE_0, tkl.i32],
        accept_token_num: tkl.Memory[S, ADDRESS_SPACE_0, tkl.i32],
        accept_index: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        cur_prob_offset_vec: tkl.Memory[S, ADDRESS_SPACE_0, tkl.i32],
        last_accepted_retrieve_idx_vec: tkl.Memory[S, ADDRESS_SPACE_0, tkl.i32],
    ):
        zero = tkw.scalar(0, tkl.i32)
        zero_f32 = tkw.scalar(0, tkl.f32)
        tkw.set_symbol(CUR_INDEX, zero)
        tkw.set_symbol(CUR_PROB_OFFSET, zero)
        tkw.set_symbol(INNER_DONE, zero)
        tkw.set_symbol(OUTER_DONE, zero)

        last_accepted_retrieve_idx = read_with_zero_offset(retrieve_index)
        coin = read_with_zero_offset(uniform_samples)

        outer_loop_cond = (J < num_speculative_tokens) | (OUTER_DONE == 1)
        inner_loop_cond = (CUR_INDEX >= 0) | (INNER_DONE == 1)

        one = tkw.scalar(1, tkl.i32)
        threshold_acc = tkw.scalar(1e-2, tkl.f32)
        threshold_acc = tkw.broadcast(threshold_acc, target_shape=[S, B, D])
        threshold_single = tkw.scalar(1e-2, tkl.f32)
        threshold_single = tkw.broadcast(threshold_single, target_shape=[S, B, D])

        @tkw.iterate(
            J,
            start=one,
            condition=outer_loop_cond,
            init_args=[zero, zero_f32, zero, last_accepted_retrieve_idx, zero, coin],
        )
        def outer_loop(
            cur_index,
            prob_acc,
            num_accepted_tokens,
            last_accepted_retrieve_idx,
            cur_prob_offset,
            coin,
        ):
            cur_index = read_with_2d_offset(retrieve_next_token)

            @tkw.iterate(
                CUR_INDEX,
                start=cur_index,
                condition=inner_loop_cond,
                init_args=[
                    cur_index,
                    prob_acc,
                    num_accepted_tokens,
                    last_accepted_retrieve_idx,
                    cur_prob_offset,
                    coin,
                ],
            )
            def inner_loop(
                cur_index,
                prob_acc,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                coin,
            ):
                draft_index = read_with_2d_offset(retrieve_index)
                draft_token_id = read_with_2d_offset(candidates)
                tkw.set_symbol(DRAFT_TOKEN_ID, draft_token_id)
                target_prob_single = read_with_3d_offset(target_probs)
                prob_acc = tkw.broadcast(prob_acc, target_shape=[S, B, D])
                prob_acc += target_prob_single

                coin_threshold = prob_acc / threshold_acc
                condition1 = coin <= coin_threshold
                condition2 = target_prob_single >= threshold_single
                condition12 = condition1 | condition2

                @tkw.conditional(condition12)
                def then():
                    write_with_3d_offset(predicts, draft_token_id)
                    tkw.write(draft_index, accept_index, elements_per_thread=1)
                    true = tkw.scalar(1, tkl.i32)
                    tkw.set_symbol(INNER_DONE, true)

                # TODO: make this work properly with not
                # condition3 = tkw.apply_expr(condition1, lambda x: sympy.Not(x))
                # condition4 = tkw.apply_expr(condition2, lambda x: sympy.Not(x))
                condition3 = coin > coin_threshold
                condition4 = target_prob_single < threshold_single
                condition34 = condition3 & condition4

                @tkw.conditional(condition34)
                def else_():
                    target_prob_reg = read_with_3d_offset_2(target_probs)
                    write_with_3d_offset(draft_probs, target_prob_reg)

                new_cur_index = tkw.select(
                    condition34,
                    tkw.broadcast(
                        read_with_2d_offset(retrieve_next_sibling),
                        target_shape=[S, B, D],
                    ),
                    tkw.broadcast(cur_index, target_shape=[S, B, D]),
                )
                tkw.set_symbol(CUR_INDEX, new_cur_index)

                new_prob_acc = tkw.select(
                    condition12,
                    tkw.broadcast(zero_f32, target_shape=[S, B, D]),
                    prob_acc,
                )

                new_accepted_tokens = tkw.select(
                    condition12,
                    tkw.broadcast(num_accepted_tokens + one, target_shape=[S, B, D]),
                    tkw.broadcast(num_accepted_tokens, target_shape=[S, B, D]),
                )

                new_last_accepted_retrieve_idx = tkw.select(
                    condition12,
                    tkw.broadcast(draft_index, target_shape=[S, B, D]),
                    tkw.broadcast(last_accepted_retrieve_idx, target_shape=[S, B, D]),
                )

                new_cur_prob_offset = tkw.select(
                    condition12,
                    tkw.broadcast(cur_index, target_shape=[S, B, D]),
                    tkw.broadcast(cur_prob_offset, target_shape=[S, B, D]),
                )
                tkw.set_symbol(CUR_PROB_OFFSET, new_cur_prob_offset)

                new_coin = tkw.select(
                    condition12,
                    tkw.broadcast(
                        read_with_2d_offset(uniform_samples), target_shape=[S, B, D]
                    ),
                    tkw.broadcast(coin, target_shape=[S, B, D]),
                )

                return (
                    new_cur_index,
                    new_prob_acc,
                    new_accepted_tokens,
                    new_last_accepted_retrieve_idx,
                    new_cur_prob_offset,
                    new_coin,
                )

            (
                cur_index,
                prob_acc,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                coin,
            ) = inner_loop
            tkw.set_symbol(CUR_INDEX, cur_index)

            @tkw.conditional(CUR_INDEX < 0)
            def then():
                true = tkw.scalar(1, tkl.i32)
                tkw.set_symbol(INNER_DONE, true)

            index_j = tkw.self_index(J, tkl.i32)
            next_value = tkw.apply_expr(index_j, lambda x: x + 1)
            tkw.set_symbol(J, next_value)

            return (
                cur_index,
                prob_acc,
                num_accepted_tokens,
                last_accepted_retrieve_idx,
                cur_prob_offset,
                coin,
            )

        (
            cur_index,
            prob_acc,
            num_accepted_tokens,
            last_accepted_retrieve_idx,
            cur_prob_offset,
            coin,
        ) = outer_loop

        tkw.write(num_accepted_tokens, accept_token_num, elements_per_thread=1)
        tkw.write(
            last_accepted_retrieve_idx,
            last_accepted_retrieve_idx_vec,
            elements_per_thread=1,
        )
        tkw.write(cur_prob_offset, cur_prob_offset_vec, elements_per_thread=1)

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
    return speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
