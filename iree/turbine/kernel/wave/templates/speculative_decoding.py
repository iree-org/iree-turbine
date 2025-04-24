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
):

    D = tkl.sym.B
    BLOCK_D = tkl.sym.BLOCK_D

    constraints = [tkw.WorkgroupConstraint(D, BLOCK_D, 0)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={D: 1},
        )
    ]

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        q: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        zero = tkl.Register[D, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=1)
        p_reg = tkw.read(p, elements_per_thread=1)
        diff = q_reg - p_reg
        relu_diff = tkw.maximum(diff, zero)
        tkw.write(relu_diff, output, elements_per_thread=1)

    hyperparams = {
        BLOCK_D: 64,
        D: batch_size,
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
    V = tkl.sym.V
    D = tkl.sym.D
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_S = tkl.sym.BLOCK_S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={B: 1, J: 1, CUR_INDEX: 1, S: 0, V: 1, D: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 0)]
    constraints += [tkw.TilingConstraint(B, BLOCK_B)]

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
        inputs={S: i, V: CUR_PROB_OFFSET, D: DRAFT_TOKEN_ID},
        outputs={S: i, V: j, D: k},
    )

    read_3d_mapping_2 = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, V: CUR_INDEX, D: DRAFT_TOKEN_ID},
        outputs={S: i, V: j, D: k},
    )

    write_1d_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={S: i, B: DRAFT_TOKEN_ID},
        outputs={S: i, B: j},
    )

    write_3d_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, V: j, D: k},
        outputs={S: i, V: CUR_INDEX, D: DRAFT_TOKEN_ID},
    )

    @tkw.wave(constraints)
    def speculative_sampling(
        a: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.f16],
        b: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.f16],
        c: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.f32],
        predicts: tkl.Memory[S * V * D, ADDRESS_SPACE_0, tkl.f32],
        uniform_samples: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.f32],
        target_probs: tkl.Memory[S, V, D, ADDRESS_SPACE_0, tkl.f32],
        draft_probs: tkl.Memory[S, V, D, ADDRESS_SPACE_0, tkl.f32],
        candidates: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_index: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_token: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        retrieve_next_sibling: tkl.Memory[S, B, ADDRESS_SPACE_0, tkl.i32],
        init_value: tkl.i32,  # type: ignore
        init_bool: tkl.i32,  # type: ignore
    ):
        # TODO: make this work properly with a Register containing value 0
        tkw.set_symbol(CUR_INDEX, init_value)
        tkw.set_symbol(CUR_PROB_OFFSET, init_value)
        # TODO: make this work properly with tkl.i1
        tkw.set_symbol(INNER_DONE, init_bool)
        tkw.set_symbol(OUTER_DONE, init_bool)

        last_accepted_retrieve_idx = tkw.read(retrieve_index, elements_per_thread=1, mapping=read_zero_offset_mapping)
        coin = tkw.read(uniform_samples, elements_per_thread=1, mapping=read_zero_offset_mapping)

        @tkw.iterate(B, init_args=[])
        def body():

            main_loop_cond = (J < num_speculative_tokens) | (OUTER_DONE == 1)
            @tkw.iterate(J, start=sympy.Integer(1), condition=main_loop_cond, init_args=[])
            def main_loop():
                cur_index = tkw.read(
                    retrieve_next_token, elements_per_thread=1, mapping=read_2d_mapping
                )

                @tkw.iterate(
                    CUR_INDEX, start=cur_index, condition=(CUR_INDEX >= 0) | (INNER_DONE == 1), init_args=[]
                )
                def repeat():
                    draft_index = tkw.read(
                        retrieve_index, elements_per_thread=1, mapping=read_2d_mapping
                    )
                    draft_token_id = tkw.read(
                        candidates, elements_per_thread=1, mapping=read_2d_mapping
                    )
                    tkw.set_symbol(DRAFT_TOKEN_ID, draft_token_id)
                    target_prob_single = tkw.read(
                        target_probs, elements_per_thread=1, mapping=read_3d_mapping
                    )
                    # TODO: make prob_acc capturable from outside of the loop
                    prob_acc = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.f32](0.0)
                    prob_acc += target_prob_single

                    # TODO: these should be defined only once outside of the loop
                    threshold_acc = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.f32](1e-2)
                    threshold_single = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.f32](1e-2)

                    coin_threshold = prob_acc / threshold_acc
                    coin = tkw.read(
                        uniform_samples, elements_per_thread=1, mapping=read_2d_mapping
                    )
                    condition1 = coin <= coin_threshold
                    condition2 = target_prob_single >= threshold_single

                    @tkw.conditional(condition1 | condition2)
                    def then():
                        prob_acc = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.f32](0.0)
                        tkw.set_symbol(CUR_PROB_OFFSET, cur_index)

                        tkw.write(
                            draft_token_id,
                            retrieve_index,
                            elements_per_thread=1,
                        #   mapping=read_2d_mapping,
                        )
                        true = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.i32](1)
                        tkw.set_symbol(INNER_DONE, true)

                    # TODO: make this work properly with not
                   #condition3 = tkw.apply_expr(condition1, lambda x: sympy.Not(x))
                   #condition4 = tkw.apply_expr(condition2, lambda x: sympy.Not(x))
                    condition3 = coin > coin_threshold
                    condition4 = target_prob_single < threshold_single

                    @tkw.conditional(condition3 & condition4)
                    def else_():
                        target_prob_reg = tkw.read(target_probs, elements_per_thread=1, mapping=read_3d_mapping_2)
                        tkw.write(target_prob_reg, draft_probs, elements_per_thread=1, mapping=write_3d_mapping)
                        cur_index = tkw.read(
                            retrieve_next_sibling,
                            elements_per_thread=1,
                            mapping=read_2d_mapping,
                        )
                        tkw.set_symbol(CUR_INDEX, cur_index)

                    tkw.set_symbol(CUR_INDEX, cur_index)

                @tkw.conditional(CUR_INDEX >= 0)
                def then():
                    true = tkw.Register[S, V, D, ADDRESS_SPACE_0, tkl.i32](1)
                    tkw.set_symbol(INNER_DONE, true)

                index_j = tkw.self_index(J, tkl.i32)
                next_value = tkw.apply_expr(index_j, lambda x: x + 1)
                tkw.set_symbol(J, next_value)

    hyperparams = {
        BLOCK_B: 1,
        B: batch_size,
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        S: 10,
        BLOCK_S: 1,
        V: 16,
        D: 20,
    }
    dynamic_symbols = []
    dynamic_symbols_map = {}
    return speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
