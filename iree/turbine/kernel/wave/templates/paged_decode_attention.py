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
from iree.turbine.kernel.wave.utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from ..symbolic_constraints import SymbolicAlias
import sympy
from enum import Enum
import math


def get_paged_decode_attention_kernels(
    shape: tuple[int],
    max_tokens: int,
    mfma_variant: MMAType,
    num_kv_splits: int,
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
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
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

    B_WAVES = 2
    M_WAVES = 2
    N_WAVES = 2
    K_WAVES = 2
    THREADS_PER_WAVE = 64
    PHASE_1_BLOCK_B = 64
    PHASE_1_ELEMS_PER_THREAD = PHASE_1_BLOCK_B // THREADS_PER_WAVE
    PHASE_1_BLOCK_N = 1

    def phase_0_constraints():
        # K1, K2 are reduction dimensions that are fixed (not distributed) so
        # they are not part of the constraints.

        constraints: list[tkw.Constraint] = []
        # U represents the number of splits of the key-value sequence.
        # U is parallelizable and is distributed across workgroups.
        constraints += [tkw.WorkgroupConstraint(U, BLOCK_U, 0)]

        # T represents the indices of the sequence tokens.
        # T is dynamic and is distributed across workgroups and is tiled.
        #
        # Each workgroup will process:
        # T = min(BLOCK, max(SEQ_LEN - WG_IDX[U] * BLOCK, 0)) tokens
        # where BLOCK = ceil(SEQ_LEN / U)
        #
        # While T and U are related to one another, since we do not know SEQ_LEN
        # we define them as symbolic aliases with different workgroup tile sizes.
        # The tile size for T is set to BLOCK_T = ceil(SEQ_LEN / U) and will also
        # be defined within the kernel.
        constraints += [tkw.WorkgroupConstraint(T, BLOCK_T, 0, primary=False)]
        constraints += [tkw.TilingConstraint(T, 1)]

        # BH is the kv-head index and is distributed across workgroups.
        # B is the query index and is distributed like BH but with a different
        # workgroup and wave tile size.
        # TODO: We will want to add a function to the workgroup constraint to
        # allow for using WG / ceil(kv_group_num, BLOCK_B) instead of just WG.
        # This can be done by adding an optional additional argument to the WorkgroupConstraint.

        constraints += [tkw.WorkgroupConstraint(BH, BLOCK_BH, 1)]
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

    # Returns the key for the given token index.
    k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={T: d0, BH: j, K2: k, K1: l},
        outputs={T: i, BH: j, K2: k, K1: l},
        dynamic_val_mappings={T: i},
    )

    # Returns the value for the given token index.
    v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={T: d0, BH: j, N: k, K2: l},
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

    @tkw.wave(get_constraints(Phase.PHASE_0))
    def phase_0(
        q: tkl.Memory[S, B, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[T, BH, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[T, BH, N, K2, ADDRESS_SPACE, tkl.f16],
        request_indices: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32],
        block_table: tkl.Memory[S, T, GLOBAL_ADDRESS_SPACE, tkl.i32],
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
        orig_seq_length = tkw.read(sequence_lengths, elements_per_thread=1)

        # The dimension T and its workgroup tile size BLOCK_T are both dynamic
        # and set below.
        tile_size = tkw.apply_expr(orig_seq_length, lambda x: sympy.ceiling(x / U))
        tkw.set_symbol(BLOCK_T, tile_size)
        seq_length = tkw.apply_expr(
            orig_seq_length,
            lambda x: sympy.Min(BLOCK_T, sympy.Max(0, x - WORKGROUP_0 * BLOCK_T)),
        )
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
            tkw.set_symbol(T, orig_seq_length)
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
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        res_max_log_sum = res_max + tkw.log2(res_sum)

        tkw.write(res_max_log_sum, output_max, elements_per_thread=1)
        tkw.write(res, output, elements_per_thread=STORE_ELEMS_PER_THREAD)

    @tkw.wave(get_constraints(Phase.PHASE_1))
    def phase_1(
        logits: tkl.Memory[U, S, N, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        logits_max: tkl.Memory[U, S, B, GLOBAL_ADDRESS_SPACE, tkl.f32],
        output: tkl.Memory[S, B, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
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
        tkw.write(
            res, output, mapping=mapping, elements_per_thread=PHASE_1_ELEMS_PER_THREAD
        )

    symbols_0 = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_BH: 1,
        BLOCK_B: shape[0] // shape[5],
        BLOCK_S: 1,
        BLOCK_U: 1,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        BH: shape[5],
        S: shape[6],
        U: num_kv_splits,
    }
    symbols_1 = dict(symbols_0)
    symbols_1[BLOCK_B] = PHASE_1_BLOCK_B
    symbols_1[BLOCK_N] = PHASE_1_BLOCK_N

    dynamic_symbols_0 = [T]
    dynamic_symbols_1 = []
    dynamic_symbols_map_0 = {T: 1}
    dynamic_symbols_map_1 = {}

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
