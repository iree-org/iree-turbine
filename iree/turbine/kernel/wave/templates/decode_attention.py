# Copyright 2024 The IREE Authors
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


def get_decode_attention_kernels(
    shape: tuple[int],
    mfma_variant: MMAType,
    use_dynamic_dims: bool = False,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
    U = tkl.sym.U
    BLOCK_U = tkl.sym.BLOCK_U

    class Phase(Enum):
        PHASE_0 = (0,)
        PHASE_1 = (1,)

    M_WAVES = 2
    N_WAVES = 2
    K_WAVES = 2
    THREADS_PER_WAVE = 64
    PHASE_1_BLOCK_M = 128
    PHASE_1_ELEMS_PER_THREAD = PHASE_1_BLOCK_M // THREADS_PER_WAVE
    PHASE_1_BLOCK_N = 1

    def phase_0_constraints():
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WaveConstraint(M, BLOCK_M / M_WAVES)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / N_WAVES)]
        constraints += [tkw.WorkgroupConstraint(K2, BLOCK_K2, 2)]
        constraints += [tkw.WaveConstraint(K2, BLOCK_K2 / K_WAVES)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 3)]
        constraints += [
            SymbolicAlias(U, K2, lambda x: sympy.ceiling(x / (BLOCK_K2 / K_WAVES)))
        ]
        vector_shapes = {B: 0}
        waves_per_block = (M_WAVES, N_WAVES, K_WAVES)
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=THREADS_PER_WAVE,
                waves_per_block=waves_per_block,
                mma_type=mfma_variant,
                vector_shapes=vector_shapes,
            )
        ]
        return constraints

    def phase_1_constraints() -> list[tkw.Constraint]:
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WaveConstraint(M, BLOCK_M)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
        constraints += [tkw.TilingConstraint(U, BLOCK_U)]
        vector_shapes = {
            B: 0,
            M: BLOCK_M,
            N: BLOCK_N,
            U: 1,
        }
        waves_per_block = (1, 1, 1)
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=THREADS_PER_WAVE,
                waves_per_block=waves_per_block,
                mma_type=mfma_variant,
                vector_shapes=vector_shapes,
            )
        ]
        return constraints

    def get_constraints(phase: Phase) -> list[tkw.Constraint]:
        if phase == Phase.PHASE_0:
            return phase_0_constraints()
        else:
            return phase_1_constraints()

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N: j, M: k},
        outputs={B: i, N: j, M: k},
    )

    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        output: tkl.Memory[U, B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output_max: tkl.Memory[U, B, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        new_acc = tkl.Register[B, N, M, tkl.f32](0.0)
        q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(k_reg, q_reg, c_reg)
        x_j = tkw.permute(acc, target_shape=[B, M, K2])
        m_j = tkw.max(x_j, init_max, dim=K2)
        e_delta_max = tkw.exp2(init_max - m_j)
        e_delta = tkw.exp2(x_j - m_j)
        e_init = init_sum * e_delta_max
        d_j = tkw.sum(e_delta, e_init, dim=K2)
        imm_f16 = tkw.cast(e_delta, tkl.f16)
        v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        acc = tkw.mma(v_reg, imm_f16, new_acc)
        res = acc / d_j
        dm_j = m_j + tkw.log2(d_j)
        tkw.write(dm_j, output_max, elements_per_thread=1)
        tkw.write(res, output, elements_per_thread=STORE_ELEMS_PER_THREAD)

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, B, N, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, B, M, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        @tkw.iterate(U, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            x_j = tkw.read(logits, elements_per_thread=PHASE_1_ELEMS_PER_THREAD)
            xm_j = tkw.read(logits_max, elements_per_thread=PHASE_1_ELEMS_PER_THREAD)
            m_j = tkw.maximum(xm_j, partial_max)
            old_scale = tkw.exp2(partial_max - m_j)
            new_scale = tkw.exp2(xm_j - m_j)
            d_j = partial_sum * old_scale + new_scale
            new_acc = acc * old_scale
            term = new_scale * x_j
            new_acc = new_acc + term
            return m_j, d_j, new_acc

        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(
            res, output, mapping=mapping, elements_per_thread=PHASE_1_ELEMS_PER_THREAD
        )

    symbols_0 = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        BLOCK_U: 1,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
    }
    symbols_0[U] = math.ceil(symbols_0[K2] / (symbols_0[BLOCK_K2] / K_WAVES))
    symbols_1 = dict(symbols_0)
    symbols_1[BLOCK_M] = PHASE_1_BLOCK_M
    symbols_1[BLOCK_N] = PHASE_1_BLOCK_N

    dynamic_symbols_0 = []
    dynamic_symbols_1 = []
    dynamic_symbols_map_0 = {}
    dynamic_symbols_map_1 = {}
    if use_dynamic_dims:
        dynamic_symbols_0 = [B, M, N, K2, U]
        for symbol in dynamic_symbols_0:
            dynamic_symbols_map_0[symbol] = symbols_0[symbol]
            del symbols_0[symbol]
            if symbol in symbols_1:
                del symbols_1[symbol]
        dynamic_symbols_1 = [x for x in dynamic_symbols_0 if x != K2]
        dynamic_symbols_map_1 = {x: dynamic_symbols_map_0[x] for x in dynamic_symbols_1}

    return (
        phase_0,
        phase_1,
        symbols_0,
        symbols_1,
        dynamic_symbols_0,
        dynamic_symbols_map_0,
        dynamic_symbols_1,
        dynamic_symbols_map_1,
    )
