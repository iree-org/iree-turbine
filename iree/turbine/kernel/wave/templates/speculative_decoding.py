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
