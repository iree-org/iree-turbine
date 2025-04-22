# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel._support.dtype import DataType
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from ..symbolic_constraints import SymbolicAlias
import sympy
from enum import Enum
import math


def get_speculative_decoding_kernel(
    batch_size: int,
    num_speculative_tokens: int,
    num_draft_tokens: int,
    d: int,
    threshold_single: float,
    threshold_acc: float,
    deterministic: bool,
):
    # Input sizes
    B = tkl.sym.B       # batch_size
    NST = tkl.sym.NST   # num_speculative_tokens
    NDT = tkl.sym.NDT   # num_draft_tokens
    D = tkl.sym.D       # vocab_size

    ONE = tkl.sym.ONE   # constant size of 1
    CUR_INDEX = tkl.sym.CUR_INDEX   # cur_index
    CUR_PROB_OFFSET = tkl.sym.CUR_PROB_OFFSET   # cur_prob_offset
    CUR_DRAFT_TOKEN_ID = tkl.sym.CUR_DRAFT_TOKEN_ID

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_NST = tkl.sym.BLOCK_NST
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    INIT_CUR_INDEX = tkl.sym.INIT_CUR_INDEX
    NEXT_CUR_INDEX = tkl.sym.NEXT_CUR_INDEX

    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]
    constraints += [tkw.TilingConstraint(B, BLOCK_B)]
    constraints += [tkw.TilingConstraint(NST, BLOCK_NST)]

    constraints += [tkw.TilingConstraint(CUR_INDEX, init_symbol=INIT_CUR_INDEX, next_symbol=NEXT_CUR_INDEX, condition=lambda x: x != -1)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, WAVES_PER_BLOCK_B),
            vector_shapes={B: 1, NST: 1},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
#   k = tkw.IndexMapping.iterator(2)

    # FIXME: check this out
    cur_index_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={NDT: CUR_INDEX},
        outputs={NDT: i},
    )

    # FIXME: check this out
    target_probs_cur_prob_offset_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NDT: CUR_PROB_OFFSET, D: CUR_DRAFT_TOKEN_ID},
        outputs={NDT: i, D: j},
    )

    target_probs_cur_index_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NDT: CUR_INDEX, D: CUR_DRAFT_TOKEN_ID},
        outputs={NDT: i, D: j},
    )

    # FIXME: this is write mapping, should we swap inputs and outputs?
    draft_probs_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NDT: CUR_INDEX, D: CUR_DRAFT_TOKEN_ID},
        outputs={NDT: i, D: j},
    )

    @tkw.wave(constraints)
    def speculative_decoding(
        predicts: tkl.Memory[B * NDT, GLOBAL_ADDRESS_SPACE, tkl.f32], # [seq_len], mutable
        accept_index: tkl.Memory[B, NST, GLOBAL_ADDRESS_SPACE, tkl.i32], # [batch_size, num_speculative_tokens], mutable
        accept_token_num: tkl.Memory[B, GLOBAL_ADDRESS_SPACE, tkl.i32], # [batch_size], mutable
        candidates: tkl.Memory[B, NDT, GLOBAL_ADDRESS_SPACE, tkl.i32],# [batch_size, num_draft_tokens]
        retrive_index: tkl.Memory[B, NDT, GLOBAL_ADDRESS_SPACE, tkl.i32], # [batch_size, num_draft_tokens]
        retrive_next_token: tkl.Memory[B, NDT, GLOBAL_ADDRESS_SPACE, tkl.i32], # [batch_size, num_draft_tokens]
        retrive_next_sibling: tkl.Memory[B, NDT, GLOBAL_ADDRESS_SPACE, tkl.i32], # [batch_size, num_draft_tokens]
        uniform_samples: tkl.Memory[B, NDT, GLOBAL_ADDRESS_SPACE, tkl.f32], # [batch_size, num_draft_tokens]
        target_probs: tkl.Memory[B, NDT, D, GLOBAL_ADDRESS_SPACE, tkl.f32], # [batch_size, num_draft_tokens, vocab_size]
        draft_probs: tkl.Memory[B, NDT, D, GLOBAL_ADDRESS_SPACE, tkl.f32], # [batch_size, num_draft_tokens, vocab_size]
    ):

        @tkw.iterate(B, init_args=[])
        def batch_iterate():
            # FIXME it seems inside this loop B dimension indexing is implicit
            prob_acc_init = tkl.Register[ONE, tkl.f32](0.0)
            cur_prob_offset_init = tkl.Register[ONE, tkl.i32](0)
            coin_init = tkw.read(uniform_samples, elements_per_thread=1)
            last_accepted_retrive_idx_init = tkw.read(retrive_index, elements_per_thread=1)
            tkw.write(last_accepted_retrive_idx_init, accept_index, elements_per_thread=1)
            num_accepted_tokens_init = tkl.Register[ONE, tkl.i32](0)
            cur_index_init = tkl.Register[ONE, tkl.i32](0)

            tkw.set_symbol(CUR_PROB_OFFSET, cur_prob_offset_init)
            tkw.set_symbol(CUR_INDEX, cur_index_init)

            @tkw.iterate(NST, init_args=[num_accepted_tokens_init])
            def speculative_token_check(num_accepted_tokens_reg: tkl.Register[ONE, tkl.i32]) -> tkl.Register[ONE, tkl.i32]:
                cur_index_init = tkw.read(retrive_index, mapping=cur_index_mapping, elements_per_thread=1)
                tkw.set_symbol(INIT_CUR_INDEX, cur_index_init)

                @tkw.iterate(CUR_INDEX, init_args=[prob_acc_init, cur_prob_offset_init, coin_init, last_accepted_retrive_idx_init, num_accepted_tokens_init, cur_index_init])
                def draft_tokens_tree_traversal(
                    prob_acc_reg: tkl.Register[ONE, tkl.f32],
                    cur_prob_offset_reg: tkl.Register[ONE, tkl.i32],
                    coin_reg: tkl.Register[ONE, tkl.f32],
                    last_accepted_retrive_idx_reg: tkl.Register[ONE, tkl.i32],
                    num_accepted_tokens_reg: tkl.Register[ONE, tkl.i32],
                    cur_index_reg: tkl.Register[ONE, tkl.i32],
                ) -> (
                    tkl.Register[ONE, tkl.f32],
                    tkl.Register[ONE, tkl.i32],
                    tkl.Register[ONE, tkl.f32],
                    tkl.Register[ONE, tkl.i32],
                    tkl.Register[ONE, tkl.i32],
                    tkl.Register[ONE, tkl.i32],
                ):
                    draft_index = tkw.read(retrive_index, mapping=cur_index_mapping, elements_per_thread=1)
                    draft_token_id = tkw.read(candidates, mapping=cur_index_mapping, elements_per_thread=1)
                    tkw.set_symbol(CUR_DRAFT_TOKEN_ID, draft_token_id)
                    target_prob_single = tkw.read(target_probs, mapping=target_probs_cur_prob_offset_mapping, elements_per_thread=1)
                    new_prob_acc_reg = prob_acc_reg + target_prob_single

                    cond = tkw.apply_expr(
                        [coin_reg, new_prob_acc_reg, target_prob_single],
                        lambda x, y, z: sympy.Or(sympy.LessThan(x, y / threshold_acc), sympy.GreaterThan(z, threshold_single)),
                    )

                    @tkw.conditional(cond)
                    def then():
                        new_prob_acc_reg = 0.0
                        new_cur_prob_offset_reg = cur_index_reg
                        new_coin_reg = tkw.read(uniform_samples, mapping=cur_index_mapping, elements_per_thread=1)
                        tkw.write(draft_token_id, predicts, mapping=predicts_write_mapping, elements_per_thread=1)
                        new_num_accepted_tokens_reg = num_accepted_tokens_reg + 1
                        tkw.write(draft_index, accept_index, mapping=accept_index_write_mapping, elements_per_thread=1)
                        new_last_accepted_retrive_idx_reg = draft_index
                        new_cur_index_reg = cur_index_reg
                        tkw.set_symbol(CUR_PROB_OFFSET, new_cur_prob_offset_reg)
                        # FIXME: revisit this once feature implemented
                        break

                    @tkw.conditional(cond)
                    def else():
                        tmp = tkw.read(target_probs, mapping=target_probs_cur_index_mapping, elements_per_thread=1)
                        tkw.write(tmp, draft_probs, mapping=draft_probs_write_mapping, elements_per_thread=1)
                        new_prob_acc_reg = prob_acc_reg
                        new_cur_prob_offset_reg = cur_prob_offset_reg
                        new_coin_reg = coin_reg
                        new_num_accepted_tokens_reg = num_accepted_tokens_reg
                        new_last_accepted_retrive_idx_reg = last_accepted_retrive_idx_reg
                        new_cur_index_reg = tkw.read(retrive_next_sibling, mapping=cur_index_mapping, elements_per_thread=1)

                    tkw.set_symbol(NEXT_CUR_INDEX, new_cur_index_reg)
                    return new_prob_acc_reg, new_cur_prob_offset_reg, new_coin_reg, new_last_accepted_retrive_idx_reg, new_num_accepted_tokens_reg, new_cur_index_reg

                _, _, _, _, num_accepted_tokens, _ = draft_tokens_tree_traversal

                @tkw.conditional(CUR_INDEX == -1)
                def then():
                    # FIXME: revisit this once feature implemented
                    return num_accepted_tokens
                    break

                return num_accepted_tokens

            tkw.write(speculative_token_check, accept_token_num, elements_per_thread=1)

            # TODO: sample from relu

        # FIXME: how to call this from speculative_decoding?
        batch_iterate

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: 4,
        MFMA_OUTPUT_ELS_PER_THREAD: 4,
        BLOCK_B: 1,
        B: batch,
        ONE: 1,
        NST: num_speculative_tokens,
        NDT: num_draft_tokens,
        D: d,
    }

    return speculative_decoding, hyperparams

