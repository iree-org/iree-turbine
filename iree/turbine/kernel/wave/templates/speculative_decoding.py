# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from dataclasses import dataclass
import math


def get_speculative_decoding_kernel(
    last_offset: int,
    num_draft_tokens: int,
    d: int,
    num_accepted_tokens: int,
    num_speculative_tokens: int,
):

    D = tkl.sym.D
    BLOCK_D = tkl.sym.BLOCK_D
    NUM_DRAFT_TOKENS = tkl.sym.NUM_DRAFT_TOKENS
    LAST_OFFSET = tkl.sym.LAST_OFFSET

    constraints = [tkw.WorkgroupConstraint(D, BLOCK_D, 0)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]
    constraints += [tkw.TilingConstraint(D, BLOCK_D)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={D: 1, NUM_DRAFT_TOKENS: LAST_OFFSET},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    prob_read_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_DRAFT_TOKENS: LAST_OFFSET, D: i},
        outputs={NUM_DRAFT_TOKENS: LAST_OFFSET, D: i},
    )

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        target_probs: tkl.Memory[NUM_DRAFT_TOKENS, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        draft_probs: tkl.Memory[NUM_DRAFT_TOKENS, D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        zero = tkl.Register[D, tkl.f32](0.0)
        q_reg = tkw.read(target_probs, mapping=prob_read_mapping, elements_per_thread=1)
        p_reg = tkw.read(draft_probs, mapping=prob_read_mapping, elements_per_thread=1)
        diff = q_reg - p_reg
        relu_diff = tkw.maximum(diff, zero)
        tkw.write(relu_diff, output, elements_per_thread=1)

    hyperparams = {
        BLOCK_D: 64,
        LAST_OFFSET: last_offset,
        NUM_DRAFT_TOKENS: num_draft_tokens,
        D: d,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return tree_speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
