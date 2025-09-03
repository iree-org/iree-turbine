# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel as tk
from iree.turbine.kernel._support.indexing import (
    index_expr,
)
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.wave_sim import wave_sim
from iree.turbine.kernel.wave.templates.conv import get_igemm_conv2d
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils.general_utils import (
    ceildiv,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    to_default_device,
    device_randn,
    device_randint,
    device_randperm,
    device_zeros,
)
from .common.utils import (
        require_e2e,
        require_cdna3,
        perf_test,
        param_bool,
        )
import torch
from torch.testing import assert_close
import pytest
import sympy
import os
import torch
import json

@require_e2e
def test_spec_dec_tail(request):
    ONE = tkl.sym.ONE
    LOOP_CARRIED_AGGREGATE = tkl.sym.LOOP_CARRIED_AGGREGATE 
    SAMPLED_ID = tkl.sym.SAMPLED_ID

    wave_size = 64

    constraints: list[tkw.Constraint] = [
            tkw.HardwareConstraint(
                threads_per_wave=wave_size,
                waves_per_block=(1, 1, 1),
                # TODO
                # vector_shapes={SEQ_LEN: 1},
                )
            ]
    constraints += [tkw.WorkgroupConstraint(SEQ_LEN, BLOCK_SEQ, 0)]
    constraints += [tkw.WaveConstraint(SEQ_LEN, BLOCK_SEQ)]
    INIT_D = tkl.sym.INIT_D
    NEXT_D = tkl.sym.NEXT_D
    constraints += [
        tkw.TilingConstraint(
                # TODO: hopefully a condition on a comparison result works. Otherwise needs to change.
            D, init_symbol=INIT_D, next_symbol=NEXT_D, condition=lambda x: x
        )
    ]
    
    # TODO: dynamic offsets on last_offset below
    k = tkw.IndexMapping.dynamic_val(0)
    b = tkw.IndexMapping.iterator(0)
    ndt = tkw.IndexMapping.iterator(1)
    d = tkw.IndexMapping.iterator(2)
    read_prob_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={B: i, NDT: k, D: d},
        outputs={B: i, N: k, D: d},
        dynamic_val_mappings={M: i, N: j},
    )



    @tkw.wave(constraints)
    def spec_dec_tail(
        predicts: tkl.Memory[B * NDT, GLOBAL_ADDRESS_SPACE, tkl.f32], # [seq_len], mutable
        target_probs: tkl.Memory[B, NDT, D, GLOBAL_ADDRESS_SPACE, tkl.f32], # [batch_size, num_draft_tokens, vocab_size]
        draft_probs: tkl.Memory[B, NDT, D, GLOBAL_ADDRESS_SPACE, tkl.f32], # [batch_size, num_draft_tokens, vocab_size]
        cur_prob_offset: tkl.i32,
        num_accepted_tokens: tkl.i32,
        num_speculative_tokens: tkl.i32,
        coin: tkl.f32
    ):
        last_offset = cur_prob_offset

        # TODO: dynamic offset on last_offset
        # q = target_probs[bx, last_offset]
        q = tkw.read(target_probs)

        # TODO: dynamic offset on last_offset
        # p = draft_probs[bx, last_offset] if num_accepted_tokens != num_speculative_tokens - 1 else torch.zeros_like(q)
        zero = tkl.Register[B, NDT, D, tkl.f32](0.0)
        p = zero

        @tkw.conditional(num_accepted_tokens != num_speculative_tokens)
        def then():
                p = tkw.read(draft_probs)

        # relu_diff = torch.relu(q - p)
        diff = q - p
        relu_diff = tkw.select(diff > zero, diff, zero)

        # sum_relu = relu_diff.sum()
        sum_relu = tkw.sum(relu_diff, dim=D)

        # TODO: Shape mismatches likely
        u = coin * sum_relu
        tkw.set_symbol(SAMPLED_ID, D - ONE)
        aggregate = tkl.Register[ONE, tkl.f32](0.0)
        zero_scalar = tkl.Register[ONE, tkl.f32](0.0)


        # TODO: I believe loop carried values not currently supported
        # for i in range(d):
        tkw.set_symbol(INIT_D, aggregate > u)

        # TODO: Can values be carried across loops like this? Should it be values in an outer scope?
        tkw.set_symbol(LOOP_CARRIED_AGGREGATE, aggregate)

        @tkw.iterate(D, init_args=[])
        def body():
          # TODO: extract val from relu_diff at self_index(D)
          #     val = relu_diff[i]

          #     if val <= 0:
          #         continue
          @tkw.conditional(val > zero_scalar)
          def then():

            # aggregate += val
            aggregate += val

            # if aggregate > u:
            @tkw.conditional(aggregate > u)
            def then():
                # sampled_id = i
                tkw.set_symbol(SAMPLED_ID, tkw.self_index(D, tkl.i32))
                
                # break
                tkw.set_symbol(NEXT_D, aggregate > u)


        # TODO: write out SAMPLED_ID properly
        tkw.write(SAMPLED_ID, predicts)


    # TODO: Parameterizations incomplete
    options = WaveCompileOptions(
        subs={
                ONE: 1,
                ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        inplace=False,
        wave_runtime=True,
    )
    set_default_run_config(options)
    test = wave_compile(options, block_adj_diff)
    print(test.asm)
    device = 'cuda'
    
    # TODO: test input
    # test(...)
