# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
import sympy


def get_speculative_decoding_kernel(
    batch_size: int,
    num_draft_tokens: int,
    vocab_size: int,
    seq_len: int,
):
    BATCH_SIZE = tkl.sym.BATCH_SIZE
    NUM_DRAFT_TOKENS = tkl.sym.NUM_DRAFT_TOKENS
    VOCAB_SIZE = tkl.sym.VOCAB_SIZE
    SEQ_LEN = tkl.sym.SEQ_LEN
    BLOCK_BATCH_SIZE = tkl.sym.BLOCK_BATCH_SIZE
    BLOCK_NUM_DRAFT_TOK = tkl.sym.BLOCK_NUM_DRAFT_TOK
    BLOCK_VOCAB_SIZE = tkl.sym.BLOCK_VOCAB_SIZE
    LAST_OFFSET = tkl.sym.LAST_OFFSET
    LAST_IDX = tkl.sym.LAST_IDX

    hyperparams = {
        BATCH_SIZE: batch_size,
        NUM_DRAFT_TOKENS: num_draft_tokens,
        VOCAB_SIZE: vocab_size,
        SEQ_LEN: seq_len,
        BLOCK_BATCH_SIZE: 1,
        BLOCK_NUM_DRAFT_TOK: 1,
        BLOCK_VOCAB_SIZE: 64,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    constraints = [tkw.WorkgroupConstraint(VOCAB_SIZE, BLOCK_VOCAB_SIZE, 0)]
    constraints += [tkw.WorkgroupConstraint(BATCH_SIZE, BLOCK_BATCH_SIZE, 1)]
    constraints += [
        tkw.WorkgroupConstraint(NUM_DRAFT_TOKENS, BLOCK_NUM_DRAFT_TOK, 1, primary=False)
    ]
    constraints += [tkw.WaveConstraint(VOCAB_SIZE, BLOCK_VOCAB_SIZE)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={BATCH_SIZE: 0, NUM_DRAFT_TOKENS: 0, VOCAB_SIZE: 64},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    target_probs_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: LAST_OFFSET, VOCAB_SIZE: k},
        outputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j, VOCAB_SIZE: k},
    )

    draft_probs_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: LAST_OFFSET, VOCAB_SIZE: k},
        outputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j, VOCAB_SIZE: k},
    )

    uniform_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: sympy.Integer(0)},
        outputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j},
    )

    output_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j},
        outputs={SEQ_LEN: LAST_IDX},
    )

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        target_probs: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, GLOBAL_ADDRESS_SPACE, tkl.f32
        ],
        draft_probs: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, GLOBAL_ADDRESS_SPACE, tkl.f32
        ],
        cur_prob_offset: tkl.Memory[BATCH_SIZE, GLOBAL_ADDRESS_SPACE, tkl.i32],
        uniform_samples: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE, tkl.f32
        ],
        last_accepted_retrive_idx_vec: tkl.Memory[
            BATCH_SIZE, GLOBAL_ADDRESS_SPACE, tkl.i32
        ],
        predicts: tkl.Memory[SEQ_LEN, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        last_offset = tkw.read(cur_prob_offset, elements_per_thread=1)
        tkw.set_symbol(LAST_OFFSET, last_offset)

        last_idx = tkw.read(last_accepted_retrive_idx_vec, elements_per_thread=1)
        tkw.set_symbol(LAST_IDX, last_idx)

        target_probs_reg = tkw.read(target_probs, mapping=target_probs_mapping)
        draft_probs_reg = tkw.read(draft_probs, mapping=draft_probs_mapping)

        # TODO: Add conditioned mask once scalar codegen is landed.
        # mask_cond = num_accepted_tokens != num_speculative_tokens_sub1
        # mask_cond = tkw.broadcast(mask_cond, target_shape=[B, N, D])
        # p_reg = tkw.select(mask_cond, p_reg, zero)

        coin = tkw.read(uniform_samples, mapping=uniform_mapping)
        diff = target_probs_reg - draft_probs_reg

        zero = tkl.Register[VOCAB_SIZE, tkl.f32](0.0)
        relu_diff = tkw.maximum(diff, zero)
        sum_relu = tkw.sum(relu_diff, dim=VOCAB_SIZE)
        cdf = tkw.cumsum(relu_diff, dim=VOCAB_SIZE)

        threshold_dist_u = tkw.broadcast(
            coin * sum_relu, target_shape=[BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE]
        )
        greater_than_u = cdf > threshold_dist_u
        # Initializing `pad_token` to the last token in the vocabulary to be default
        # and within bounds.
        pad_token = tkl.Register[BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, tkl.i32](
            VOCAB_SIZE - 1
        )
        token_idx = tkl.Register[BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, tkl.i32](
            THREAD_0
        )

        # TODO: We can implement with `ballot(greater_than_u)` and early exit
        #       /return d-1 if output are all zeros.
        # If no valid token is found, use d-1 token.
        valid_lane_token_idx = tkw.select(greater_than_u, token_idx, pad_token)
        min_valid_token_idx = tkw.min(valid_lane_token_idx, dim=VOCAB_SIZE)
        tkw.write(min_valid_token_idx, predicts, mapping=output_mapping)

    return tree_speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map


def get_speculative_sampling_kernel(
    batch_size: int,
    num_speculative_tokens: int,
    threshold_acc: float,
    threshold_single: float,
    num_draft_tokens: int,
    vocab_size: int,
    seq_len: int,
):
    CUR_INDEX = sympy.Symbol("CUR_INDEX")
    J = sympy.Symbol("J")
    BATCH_SIZE = tkl.sym.BATCH_SIZE
    NUM_DRAFT_TOKENS = tkl.sym.NUM_DRAFT_TOKENS
    NUM_SPECULATIVE_TOKENS = tkl.sym.NUM_SPECULATIVE_TOKENS
    VOCAB_SIZE = tkl.sym.VOCAB_SIZE
    SEQ_LEN = tkl.sym.SEQ_LEN
    BLOCK_BATCH_SIZE = tkl.sym.BLOCK_BATCH_SIZE
    BLOCK_NUM_DRAFT_TOK = tkl.sym.BLOCK_NUM_DRAFT_TOK
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    GLOBAL_ADDRESS_SPACE_0 = tkl.sym.GLOBAL_ADDRESS_SPACE

    hyperparams = {
        BLOCK_NUM_DRAFT_TOK: 1,
        NUM_DRAFT_TOKENS: num_draft_tokens,
        NUM_SPECULATIVE_TOKENS: num_speculative_tokens,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        GLOBAL_ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        BATCH_SIZE: batch_size,
        BLOCK_BATCH_SIZE: 1,
        VOCAB_SIZE: vocab_size,
        SEQ_LEN: seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={
                NUM_DRAFT_TOKENS: num_draft_tokens,
                J: 0,
                CUR_INDEX: 0,
                BATCH_SIZE: 0,
                VOCAB_SIZE: vocab_size,
                NUM_SPECULATIVE_TOKENS: num_speculative_tokens,
            },
        )
    ]
    # we distribute BATCH_SIZE along WG dim_1 because of the mapping constraint.
    # BATCH_SIZE can be lesser than num of threads.
    constraints += [tkw.WorkgroupConstraint(BATCH_SIZE, BLOCK_BATCH_SIZE, 1)]
    constraints += [tkw.TilingConstraint(CUR_INDEX)]
    constraints += [tkw.TilingConstraint(J)]

    CUR_PROB_OFFSET = tkl.sym.CUR_PROB_OFFSET
    DRAFT_TOKEN_ID = tkl.sym.DRAFT_TOKEN_ID
    NUM_ACCEPTED_TOKENS = tkl.sym.NUM_ACCEPTED_TOKENS
    LAST_ACCEPTED_RETRIEVE_IDX = tkl.sym.LAST_ACCEPTED_RETRIEVE_IDX

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    # read mappings
    read_mapping_2d_to_1d = tkw.IndexMapping(
        num_iterators=1,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: CUR_INDEX},
        outputs={BATCH_SIZE: i},
    )

    read_mapping_3d_to_1d = tkw.IndexMapping(
        num_iterators=1,
        inputs={
            BATCH_SIZE: i,
            NUM_DRAFT_TOKENS: CUR_PROB_OFFSET,
            VOCAB_SIZE: DRAFT_TOKEN_ID,
        },
        outputs={BATCH_SIZE: i},
    )

    read_mapping_3d_to_3d = tkw.IndexMapping(
        num_iterators=3,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: CUR_INDEX, VOCAB_SIZE: DRAFT_TOKEN_ID},
        outputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j, VOCAB_SIZE: k},
    )

    read_mapping_zero_offset_2d_to_1d = tkw.IndexMapping(
        num_iterators=1,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: sympy.Integer(0)},
        outputs={BATCH_SIZE: i},
    )

    # write mappings
    write_mapping_1d_to_2d = tkw.IndexMapping(
        num_iterators=1,
        inputs={BATCH_SIZE: i},
        outputs={
            BATCH_SIZE: i,
            NUM_SPECULATIVE_TOKENS: NUM_ACCEPTED_TOKENS,
        },
    )

    write_mapping_1d_to_1d = tkw.IndexMapping(
        num_iterators=1,
        inputs={BATCH_SIZE: i},
        outputs={SEQ_LEN: LAST_ACCEPTED_RETRIEVE_IDX},
    )

    write_mapping_3d_to_3d = tkw.IndexMapping(
        num_iterators=3,
        inputs={BATCH_SIZE: i, NUM_DRAFT_TOKENS: j, VOCAB_SIZE: k},
        outputs={
            BATCH_SIZE: i,
            NUM_DRAFT_TOKENS: CUR_INDEX,
            VOCAB_SIZE: DRAFT_TOKEN_ID,
        },
    )

    write_mapping_zero_offset_1d_to_2d = tkw.IndexMapping(
        num_iterators=1,
        inputs={BATCH_SIZE: i},
        outputs={
            BATCH_SIZE: i,
            NUM_SPECULATIVE_TOKENS: sympy.Integer(0),
        },
    )

    # read/write helper functions
    def read_2d_into_1d(x):
        return tkw.read(x, elements_per_thread=1, mapping=read_mapping_2d_to_1d)

    def read_3d_into_1d(x):
        return tkw.read(x, elements_per_thread=1, mapping=read_mapping_3d_to_1d)

    def read_3d_into_3d(x):
        return tkw.read(x, elements_per_thread=1, mapping=read_mapping_3d_to_3d)

    def read_with_zero_offset_2d_into_1d(memory):
        return tkw.read(
            memory, elements_per_thread=1, mapping=read_mapping_zero_offset_2d_to_1d
        )

    def write_2d_into_1d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_1d_to_2d)

    def write_1d_into_1d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_1d_to_1d)

    def write_3d_into_3d(x, y):
        return tkw.write(x, y, elements_per_thread=1, mapping=write_mapping_3d_to_3d)

    def write_with_zero_offset_1d_into_2d(x, y):
        return tkw.write(
            x, y, elements_per_thread=1, mapping=write_mapping_zero_offset_1d_to_2d
        )

    # Kernel.
    # =================================================================================
    @tkw.wave(constraints)
    def speculative_sampling(
        uniform_samples: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE_0, tkl.f32
        ],
        target_probs: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, GLOBAL_ADDRESS_SPACE_0, tkl.f32
        ],
        draft_probs: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, VOCAB_SIZE, GLOBAL_ADDRESS_SPACE_0, tkl.f32
        ],
        candidates: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE_0, tkl.i32
        ],
        retrieve_index: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE_0, tkl.i32
        ],
        retrieve_next_token: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE_0, tkl.i32
        ],
        retrieve_next_sibling: tkl.Memory[
            BATCH_SIZE, NUM_DRAFT_TOKENS, GLOBAL_ADDRESS_SPACE_0, tkl.i32
        ],
        # Outputs
        predicts: tkl.Memory[SEQ_LEN, GLOBAL_ADDRESS_SPACE_0, tkl.i32],
        accept_token_num: tkl.Memory[
            BATCH_SIZE,
            GLOBAL_ADDRESS_SPACE_0,
            tkl.i32,
        ],
        accept_index: tkl.Memory[
            BATCH_SIZE,
            NUM_SPECULATIVE_TOKENS,
            GLOBAL_ADDRESS_SPACE_0,
            tkl.i32,
        ],
        cur_prob_offset_vec: tkl.Memory[
            BATCH_SIZE,
            GLOBAL_ADDRESS_SPACE_0,
            tkl.i32,
        ],
        last_accepted_retrieve_idx_vec: tkl.Memory[
            BATCH_SIZE,
            GLOBAL_ADDRESS_SPACE_0,
            tkl.i32,
        ],
    ):
        one = tkw.Register[BATCH_SIZE, tkl.i32](1)
        zero = tkw.Register[BATCH_SIZE, tkl.i32](0)
        zero_f32 = tkw.Register[BATCH_SIZE, tkl.f32](0.0)

        threshold_acc_reg = tkw.Register[BATCH_SIZE, tkl.f32](threshold_acc)
        threshold_single_reg = tkw.Register[BATCH_SIZE, tkl.f32](threshold_single)

        outer_loop_condition = (J < num_speculative_tokens) & (
            sympy.Eq(GET_ITER_ARG(6), 0)
        )
        inner_loop_condition = (CUR_INDEX >= 0) & (sympy.Eq(GET_ITER_ARG(6), 0))

        coin = read_with_zero_offset_2d_into_1d(uniform_samples)
        last_accepted_retrieve_idx = read_with_zero_offset_2d_into_1d(retrieve_index)
        write_with_zero_offset_1d_into_2d(last_accepted_retrieve_idx, accept_index)

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
            cur_index = read_2d_into_1d(retrieve_next_token)

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
                draft_index = read_2d_into_1d(retrieve_index)
                draft_token_id = read_2d_into_1d(candidates)
                tkw.set_symbol(DRAFT_TOKEN_ID, draft_token_id)
                tkw.set_symbol(CUR_PROB_OFFSET, cur_prob_offset)
                target_prob_single = read_3d_into_1d(target_probs)

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
                    cur_index,
                    cur_prob_offset,
                )

                # Update coin.
                coin = tkw.select(
                    condition,
                    read_2d_into_1d(uniform_samples),
                    coin,
                )

                @tkw.conditional(condition)
                def then_():
                    write_1d_into_1d(draft_token_id, predicts)
                    write_2d_into_1d(draft_index, accept_index)

                @tkw.conditional(not_condition)
                def else_():
                    target_prob = read_3d_into_3d(target_probs)
                    write_3d_into_3d(target_prob, draft_probs)

                # Update last_accepted_retrieve_idx.
                last_accepted_retrieve_idx = tkw.select(
                    condition,
                    draft_index,
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
                    read_2d_into_1d(retrieve_next_sibling),
                    cur_index,
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
