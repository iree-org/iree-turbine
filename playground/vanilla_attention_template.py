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
from iree.turbine.kernel.wave.templates.attention_common import AttentionShape
from dataclasses import dataclass


def get_vanilla_attention_kernel(shape: AttentionShape, mfma_variant: MMAType,
                                 dynamic_dims: bool):
    # RPE
    ZERO = tkl.sym.ZERO
    OFFSET = tkl.sym.OFFSET

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
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0)
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(4, 1, 1),
            mma_type=mfma_variant[1],
            vector_shapes={
                B: 0,
                M: Mvec,
                N: Nvec
            },
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(num_iterators=3,
                               inputs={
                                   B: i,
                                   N: j,
                                   M: k
                               },
                               outputs={
                                   B: i,
                                   M: k,
                                   N: j
                               })

    offset_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={K2: i + OFFSET},
        outputs={K2: i},
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        # TODO: double check whether MAX_CONTEXT_LENGTH relates to K2 actually
        # TODO: there is a memory logical layout different from the physical one
        # so we could have different sizes at runtime but still use the same
        # symbol for the shape.
        rpe: tkl.Memory[K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, M, tkl.f32](0.0)
        init_max = tkl.Register[B, M, tkl.f32](-1e6)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, M, tkl.f32],
            partial_sum: tkl.Register[B, M, tkl.f32],
            acc: tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])

            # T5 RPE adds attention bias pre softmax. When fusing into flash
            # attention variant, add before the max and the partial softmax.
            i = tkw.self_index(M, tkl.i64, elements_per_thread=1)
            i = tkw.broadcast(i, target_shape=[K2])
            j = tkw.self_index(K2, tkl.i64, elements_per_thread=1)

            # TODO: some errors below.
            # ZERO = i - i
            # # TODO: tkw.maximum/minimum to support 0 instead of requiring a ZERO
            # # symbol: ValueError: Expected an fx.Node but got <class 'int'>
            # # TODO: Expected an fx.Node but got <class 'sympy.core.symbol.Symbol'>
            # idx = tkw.maximum(i - j, ZERO)
            # # TODO(ntv): add tkw.minimum
            # # TODO(ntv): may actually need MAX_CONTEXT_LENGTH here
            # # idx = tkw.minimum(idx, K2)
            # TODO: may need adjustements depending on how we want to do
            # bucketing; atm it is bucketing of size 1.

            idx = i - j
            tkw.set_symbol(OFFSET, idx)  # offset will have shape [K2]

            # Note: this is a read indirect with an OFFSET via offset_mapping.
            rpe_reg = tkw.read(rpe,
                               mapping=offset_mapping,
                               elements_per_thread=LOAD_ELEMS_PER_THREAD_QK)
            x_j = x_j + rpe_reg

            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD_PV)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        tkw.write(res,
                  c,
                  mapping=mapping,
                  elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK:
        get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV:
        get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD:
        get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_B: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape.num_query_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: shape.kv_seq_len,
    }

    dynamic_symbols = []
    dynamic_symbols_map = {}
    if dynamic_dims:
        dynamic_symbols_map[M] = hyperparams[M]
        dynamic_symbols_map[N] = hyperparams[N]
        dynamic_symbols_map[B] = hyperparams[B]
        dynamic_symbols_map[K2] = hyperparams[K2]
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    return base_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
