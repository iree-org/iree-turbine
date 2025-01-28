# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
import sympy
from enum import Enum
from collections import namedtuple
import math

paged_decode_attention_shape = namedtuple(
    "paged_decode_attention_shape",
    [
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "head_size_kv",
        "block_size",
        "num_seqs",
        "kv_lens",
    ],
)


def get_paged_decode_attention_kernels(
    shape: paged_decode_attention_shape,
    mfma_variant: MMAType,
    num_kv_splits: int,
    k_shape: tuple[int],
    v_shape: tuple[int],
    block_table_shape: tuple[int],
):
    # Input sizes
    T = tkl.sym.T
    S = tkl.sym.S
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    U = tkl.sym.U
    BH = tkl.sym.BH
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BH = tkl.sym.BLOCK_BH
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_U = tkl.sym.BLOCK_U
    BLOCK_T = tkl.sym.BLOCK_T
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    class Phase(Enum):
        PHASE_0 = (0,)
        PHASE_1 = (1,)

    THREADS_PER_WAVE = 64
    PHASE_1_BLOCK_B = 64
    PHASE_1_ELEMS_PER_THREAD = PHASE_1_BLOCK_B // THREADS_PER_WAVE
    PHASE_1_BLOCK_N = 1
    HEAD_BLOCK_SIZE = 16
    head_ratio = shape.num_query_heads // shape.num_kv_heads
    B_WAVES = 1

    # T represents the indices of the sequence tokens.
    # T is dynamic and is distributed across workgroups and is tiled.
    iter_count = sympy.Piecewise(
        (sympy.ceiling(T / U), WORKGROUP_0 < sympy.Mod(T, U)),
        (sympy.floor(T / U), True),
    )

    def phase_0_constraints():
        # K1, K2 are reduction dimensions that are fixed (not distributed) so
        # they are not part of the constraints.

        constraints: list[tkw.Constraint] = []
        # U represents the number of splits of the key-value sequence.
        # U is parallelizable and is distributed across workgroups.
        constraints += [tkw.WorkgroupConstraint(U, BLOCK_U, 0)]

        wg_func = lambda wg: wg * sympy.floor(T / U) + sympy.Min(wg, sympy.Mod(T, U))
        constraints += [
            tkw.WorkgroupConstraint(T, BLOCK_T, 0, apply_fn=wg_func, primary=False)
        ]
        constraints += [tkw.TilingConstraint(T, 1, iters=iter_count)]

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
            tkw.WorkgroupConstraint(BH, BLOCK_BH, 1, apply_fn=wg_func_2, iters=count)
        ]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 1, primary=False)]
        constraints += [tkw.WaveConstraint(B, BLOCK_B / B_WAVES)]

        constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 2)]

        vector_shapes = {BH: 0, T: 0, S: 0, U: 1}
        waves_per_block = (1, B_WAVES, 1)
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
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(B, BLOCK_B, 0)]
        constraints += [tkw.WaveConstraint(B, BLOCK_B)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N)]
        constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 2)]
        constraints += [tkw.TilingConstraint(U, BLOCK_U)]
        vector_shapes = {
            S: 0,
            B: BLOCK_B,
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
    l = tkw.IndexMapping.iterator(3)
    d0 = tkw.IndexMapping.dynamic_val(0)

    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={S: i, B: j, N: k},
        outputs={S: i, B: j, N: k},
    )

    K2_dim = k_shape[1]
    # Returns the key for the given token index.
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={T: d0 // K2_dim, BH: j, K2: d0 % K2_dim, K1: l},
        outputs={T: i, BH: j, K2: k, K1: l},
        dynamic_val_mappings={T: i},
    )

    # Returns the value for the given token index.
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={T: d0 // K2_dim, BH: j, N: k, K2: d0 % K2_dim},
        outputs={T: i, BH: j, N: k, K2: l},
        dynamic_val_mappings={T: i},
    )

    # Returns token indices into the k-v cache for the given sequence (d0).
    block_table_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: d0, T: j},
        outputs={S: i, T: j},
        dynamic_val_mappings={S: i},
    )

    k_layout = tkl.MemoryLayout(shape=k_shape)
    v_layout = tkl.MemoryLayout(shape=v_shape)
    block_table_layout = tkl.MemoryLayout(shape=block_table_shape)

    # The kv-cache layout here is (SEQ, HEADS, HEAD_DIM).
    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[T, K2, BH, K1, ADDRESS_SPACE, tkl.f16, k_layout],
        v: tkl.Memory[T, K2, BH, N, ADDRESS_SPACE, tkl.f16, v_layout],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        block_table: tkl.Memory[
            S, T, GLOBAL_ADDRESS_SPACE, tkl.i32, block_table_layout
        ],
        output: tkl.Memory[U, S, N, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        # =========================================================================
        # Query has shape [NUM_SEQS, NUM_HEADS, HEAD_DIM]
        # Key has shape [NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
        # Value has shape [NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE]
        #                 (TODO: This is a transposed version of the original)
        # Sequence lengths has shape [NUM_SEQS]
        # Request indices has shape [NUM_SEQS]
        # Block table has shape [NUM_SEQS, MAX_KV_SEQ_LEN]
        # Output has shape [NUM_KV_SPLITS, NUM_SEQS, NUM_HEADS, HEAD_DIM]
        # =========================================================================

        init_max = tkl.Register[S, B, tkl.f32](-1e6)
        init_sum = tkl.Register[S, B, tkl.f32](0.0)
        new_acc = tkl.Register[S, N, B, tkl.f32](0.0)

        # The request index is used to load the appropriate entries from the block table.
        req_index = tkw.read(request_indices, elements_per_thread=1)
        # The sequence length is used to control the bounds of the loop over T.
        seq_length = tkw.read(sequence_lengths, elements_per_thread=1)
        tkw.set_symbol(T, seq_length)

        # TODO: Add if statement here in cases where T is 0 to avoid writing nans for the output.
        # While the for loop will be skipped, the calculations and writes outside the for
        # loop will still be executed.

        @tkw.reduction(T, init_args=[init_max, init_sum, new_acc])
        def loop(
            partial_max: tkl.Register[S, B, tkl.f32],
            partial_sum: tkl.Register[S, B, tkl.f32],
            acc: tkl.Register[S, N, B, tkl.f32],
        ):
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            block_indices = tkw.read(
                block_table,
                elements_per_thread=1,
                mapping=block_table_mapping,
                mapping_dynamic_vals=(req_index,),
            )
            k_reg = tkw.read(
                k,
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
                mapping=k_mapping,
                mapping_dynamic_vals=(block_indices,),
            )
            imm_reg = tkl.Register[S, K2, B, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[S, B, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v,
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
                mapping=v_mapping,
                mapping_dynamic_vals=(block_indices,),
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = loop

        @tkw.conditional(iter_count > 0)
        def then():
            reciprocal_sum = tkw.reciprocal(res_sum)
            res = res_mm * reciprocal_sum
            res_max_log_sum = res_max + tkw.log2(res_sum)

            tkw.write(res_max_log_sum, output_max, elements_per_thread=1)
            tkw.write(res, output, elements_per_thread=STORE_ELEMS_PER_THREAD)

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, S, N, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        c_reg = tkl.Register[S, B, N, tkl.f32](0.0)
        init_sum = tkl.Register[S, B, tkl.f32](0.0)
        init_max = tkl.Register[S, B, tkl.f32](-1e6)

        @tkw.reduction(U, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[S, B, tkl.f32],
            partial_sum: tkl.Register[S, B, tkl.f32],
            acc: tkl.Register[S, B, N, tkl.f32],
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
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_BH: 1,
        BLOCK_B: HEAD_BLOCK_SIZE,
        BLOCK_S: 1,
        BLOCK_U: 1,
        B: shape.num_query_heads,
        M: 1,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.block_size,
        BH: shape.num_kv_heads,
        S: shape.num_seqs,
        U: num_kv_splits,
    }
    symbols_1 = dict(symbols_0)
    symbols_1[BLOCK_B] = PHASE_1_BLOCK_B
    symbols_1[BLOCK_N] = PHASE_1_BLOCK_N

    return (
        phase_0,
        phase_1,
        symbols_0,
        symbols_1,
    )
