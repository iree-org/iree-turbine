# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
import sympy
from enum import Enum
import math
from .attention_common import *


def get_gqa_decode_attention_kernels(
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    num_kv_splits: int,
    k_shape: tuple[int],
    v_shape: tuple[int],
    layer_scaling: float = None,
):
    # Input sizes
    B = tkl.sym.B  # Num seqs
    U = tkl.sym.U  # Num splits
    SPLIT_OFF = tkl.sym.SPLIT_OFF
    SPLIT_LEN = tkl.sym.SPLIT_LEN
    # Workgroup tile sizes
    BLOCK_U = tkl.sym.BLOCK_U
    BLOCK_B = tkl.sym.BLOCK_B
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    class Phase(Enum):
        PHASE_0 = (0,)
        PHASE_1 = (1,)

    THREADS_PER_WAVE = 64
    PHASE_1_BLOCK_B = 64
    PHASE_1_ELEMS_PER_THREAD = PHASE_1_BLOCK_B // THREADS_PER_WAVE
    PHASE_1_BLOCK_N = 1
    HEAD_BLOCK_SIZE = 32
    head_ratio = shape.num_query_heads // shape.num_kv_heads
    seq_len_per_split = math.ceil(shape.kv_seq_len / num_kv_splits)
    B_WAVES = 1
    LOG2E = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E

    def phase_0_constraints():
        # K1, K2 are reduction dimensions that are fixed (not distributed) so
        # they are not part of the constraints.

        constraints: list[tkw.Constraint] = []
        # U represents the number of splits of the key-value sequence.
        # U is parallelizable and is distributed across workgroups.
        constraints += [tkw.WorkgroupConstraint(U, BLOCK_U, 2)]
        constraints += [
            tkw.TilingConstraint(
                N_KV,
                BLOCK_N_KV,
                iters=sympy.ceiling(SPLIT_LEN / BLOCK_N_KV),
                start=SPLIT_OFF,
            )
        ]

        # BH is the kv-head index and is distributed across workgroups.
        # B is the query index and is distributed like BH but with a different
        # workgroup and wave tile size.

        # For GQA, the number of query heads >> number of kv heads. So we launch
        # workgroups where each workgroup processes HEAD_BLOCK_SIZE query heads
        # as this allows us to use MMA for the attention computation. While
        # each workgroup processes HEAD_BLOCK_SIZE query heads, it only processes
        # one kv head. So we need to specify an apply_func to determine the
        # appropriate kv head index.

        wg_func_2 = lambda wg: wg // math.ceil(head_ratio / HEAD_BLOCK_SIZE)
        count = shape.num_query_heads // min(HEAD_BLOCK_SIZE, head_ratio)
        constraints += [
            tkw.WorkgroupConstraint(
                H_KV, BLOCK_H_KV, 1, apply_fn=wg_func_2, iters=count
            )
        ]
        constraints += [tkw.WorkgroupConstraint(H_Q, BLOCK_H_Q, 1, primary=False)]
        constraints += [tkw.WaveConstraint(H_Q, BLOCK_H_Q / B_WAVES)]

        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]

        vector_shapes = {H_KV: 0, B: 0, U: 1}
        waves_per_block = (1, B_WAVES, 1)
        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=THREADS_PER_WAVE,
                waves_per_block=waves_per_block,
                mma_type=mfma_variant[1],
                vector_shapes=vector_shapes,
            )
        ]
        return constraints

    def phase_1_constraints() -> list[tkw.Constraint]:
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(H_Q, BLOCK_H_Q, 0)]
        constraints += [tkw.WaveConstraint(H_Q, BLOCK_H_Q)]
        constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
        constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
        constraints += [tkw.TilingConstraint(U, BLOCK_U)]
        vector_shapes = {
            B: 0,
            H_Q: BLOCK_H_Q,
            D_KV: BLOCK_D_KV,
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
    l = tkw.IndexMapping.iterator(3)

    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, H_Q: j, D_KV: k},
        outputs={B: i, H_Q: j, D_KV: k},
    )

    # Returns the key for the given token index.
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j, N_KV: k, D_Q: l},
        outputs={B: i, H_KV: j, N_KV: k, D_Q: l},
    )

    # Returns the value for the given token index.
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H_KV: j, D_KV: k, N_KV: l},
        outputs={B: i, H_KV: j, D_KV: k, N_KV: l},
    )

    k_layout = tkl.MemoryLayout(shape=k_shape)
    v_layout = tkl.MemoryLayout(shape=v_shape)

    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[B, H_Q, D_Q, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, N_KV, H_KV, D_Q, ADDRESS_SPACE, tkl.f16, k_layout],
        v: tkl.Memory[B, N_KV, H_KV, D_KV, ADDRESS_SPACE, tkl.f16, v_layout],
        output: tkl.Memory[U, B, D_KV, H_Q, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output_max: tkl.Memory[U, B, H_Q, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):

        init_max = tkl.Register[B, H_Q, tkl.f32](-1e6)
        init_sum = tkl.Register[B, H_Q, tkl.f32](0.0)
        new_acc = tkl.Register[B, D_KV, H_Q, tkl.f32](0.0)
        layer_scaling_reg = tkl.Register[B, H_Q, N_KV, tkl.f32](layer_scaling)
        zero = tkl.Register[H_Q, N_KV, tkl.f32](0.0)
        neg_infinity = tkl.Register[H_Q, N_KV, tkl.f32](-1e6)

        @tkw.iterate(N_KV, init_args=[init_max, init_sum, new_acc])
        def loop(
            partial_max: tkl.Register[B, H_Q, tkl.f32],
            partial_sum: tkl.Register[B, H_Q, tkl.f32],
            acc: tkl.Register[B, D_KV, H_Q, tkl.f32],
        ):
            q_reg = tkw.read(q)
            k_reg = tkw.read(k, mapping=k_mapping)
            imm_reg = tkl.Register[B, N_KV, H_Q, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H_Q, N_KV])
            k2_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(k2_index, lambda x: x < (SPLIT_OFF + SPLIT_LEN))
            mask = tkw.broadcast(mask, target_shape=[H_Q, N_KV])
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, zero, neg_infinity)
            x_j = x_j + bias
            x_j = x_j * layer_scaling_reg
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, mapping=v_mapping)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = loop

        @tkw.conditional(SPLIT_LEN > 0)
        def then():
            reciprocal_sum = tkw.reciprocal(res_sum)
            res = res_mm * reciprocal_sum
            res_max_log_sum = res_max + tkw.log2(res_sum)

            tkw.write(res_max_log_sum, output_max)
            tkw.write(res, output)

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, B, D_KV, H_Q, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, B, H_Q, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[B, H_Q, D_KV, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        c_reg = tkl.Register[B, H_Q, D_KV, tkl.f32](0.0)
        init_sum = tkl.Register[B, H_Q, tkl.f32](0.0)
        init_max = tkl.Register[B, H_Q, tkl.f32](-1e6)

        @tkw.iterate(U, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H_Q, tkl.f32],
            partial_sum: tkl.Register[B, H_Q, tkl.f32],
            acc: tkl.Register[B, H_Q, D_KV, tkl.f32],
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
        res_f16 = tkw.cast(res, tkl.f16)
        tkw.write(
            res_f16,
            output,
            mapping=mapping,
            elements_per_thread=PHASE_1_ELEMS_PER_THREAD,
        )

    symbols_0 = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_H_KV: 1,
        BLOCK_H_Q: HEAD_BLOCK_SIZE,
        BLOCK_B: 1,
        BLOCK_U: 1,
        BLOCK_N_KV: 32,
        H_Q: shape.num_query_heads,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
        N_KV: shape.kv_seq_len,
        H_KV: shape.num_kv_heads,
        B: shape.num_seqs,
        U: num_kv_splits,
        SPLIT_OFF: WORKGROUP_2 * seq_len_per_split,
        SPLIT_LEN: sympy.Min(N_KV, (WORKGROUP_2 + 1) * seq_len_per_split) - SPLIT_OFF,
    }
    # Simplify the symbols by substituting the values of the symbols.
    for key, value in symbols_0.items():
        if isinstance(value, sympy.Expr):
            symbols_0[key] = value.subs(symbols_0)

    symbols_1 = dict(symbols_0)
    symbols_1[BLOCK_H_Q] = PHASE_1_BLOCK_B
    symbols_1[BLOCK_D_KV] = PHASE_1_BLOCK_N

    return (
        phase_0,
        phase_1,
        symbols_0,
        symbols_1,
    )
