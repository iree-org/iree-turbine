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


def get_speculative_decoding_kernel(
    batch_size: int,
):

    D = tkl.sym.B
    BLOCK_D = tkl.sym.BLOCK_D
    CDF_ITER = tkl.sym.CDF_ITER
    CDF_THRESH = tkl.sym.CDF_THRESH

    constraints = [tkw.WorkgroupConstraint(D, BLOCK_D, 0)]
    constraints += [tkw.WaveConstraint(D, BLOCK_D)]
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={D: 64},
        )
    ]

    @tkw.wave(constraints)
    def tree_speculative_sampling(
        q: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        coin: tkl.f32,
        relu_diff_out: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
        u_out: tkl.Memory[D, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        zero = tkl.Register[D, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=1)
        p_reg = tkw.read(p, elements_per_thread=1)
        diff = q_reg - p_reg
        relu_diff = tkw.maximum(diff, zero)
        sum_relu = tkw.sum(relu_diff, dim=D)
        out = tkw.broadcast(coin * sum_relu, target_shape=[D])
        tkw.write(relu_diff, relu_diff_out, elements_per_thread=1)
        tkw.write(out, u_out, elements_per_thread=1)

    hyperparams = {
        BLOCK_D: 64,
        D: batch_size,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}

    return tree_speculative_sampling, hyperparams, dynamic_symbols, dynamic_symbols_map
