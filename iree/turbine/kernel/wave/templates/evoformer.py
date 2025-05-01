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


def get_evoformer_kernel(
    batch: tuple[int, int],
    n: tuple[int, int],
    kv_seq_len: tuple[int, int],
    heads: tuple[int, int],
    head_dim: tuple[int, int],
    q_seq_len: tuple[int, int],
    v_dim: tuple[int, int],
    mfma_variant: MMAType,
    datatype: DataType,
):
    assert datatype in [tkl.f16, tkl.bf16], f"Unsupported datatype: {datatype}"

    # Input sizes
    B = tkl.sym.B
    BN = tkl.sym.BN
    M = tkl.sym.M
    H = tkl.sym.H
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BN = tkl.sym.BLOCK_BN
    BLOCK_H = tkl.sym.BLOCK_H
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    ratio_m = 2
    ratio_n = 1
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 3)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 4)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / ratio_n)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_m, ratio_n, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, BN: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    m = tkw.IndexMapping.iterator(4)
    # [B, BN, M, H, K1] -> [B, BN, H, M, K1]
    q_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, M: l, K1: m},
        outputs={B: i, BN: j, H: k, M: l, K1: m},
    )
    # [B, BN, K2, H, K1] -> [B, BN, H, K2, K1]
    k_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, K2: l, K1: m},
        outputs={B: i, BN: j, H: k, K2: l, K1: m},
    )
    # [B, BN, N, H, K2] -> [B, BN, H, N, K2]
    v_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, K2: m},
        outputs={B: i, BN: j, H: k, N: l, K2: m},
    )
    # [B, BN, H, N, M] -> [B, BN, M, H, N]
    o_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, M: m},
        outputs={B: i, BN: j, H: k, N: l, M: m},
    )

    @tkw.wave(constraints)
    def evoformer_fwd(
        q: tkl.Memory[B, BN, M, H, K1, GLOBAL_ADDRESS_SPACE, datatype],
        k: tkl.Memory[B, BN, K2, H, K1, ADDRESS_SPACE, datatype],
        v: tkl.Memory[B, BN, N, H, K2, ADDRESS_SPACE, datatype],
        mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, datatype],
        bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, datatype],
        c: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, datatype],
    ):
        c_reg = tkl.Register[B, BN, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, BN, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, BN, H, M, tkl.f32](-1e6)

        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, BN, H, M, tkl.f32],
            partial_sum: tkl.Register[B, BN, H, M, tkl.f32],
            acc: tkl.Register[B, BN, H, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, BN, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(
                q, mapping=q_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                q_reg = tkw.cast(tkw.cast(q_reg, tkl.f32), tkl.f16)
            k_reg = tkw.read(
                k, mapping=k_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                k_reg = tkw.cast(tkw.cast(k_reg, tkl.f32), tkl.f16)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, BN, H, M, K2])
            mask_reg = tkw.read(mask, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_mask_reg = tkw.cast(mask_reg, tkl.f32)
            y_j = x_j + casted_mask_reg
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_bias_reg = tkw.cast(bias_reg, tkl.f32)
            z_j = y_j + casted_bias_reg
            m_j = tkw.max(z_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(z_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v, mapping=v_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            if datatype == tkl.bf16:
                v_reg = tkw.cast(tkw.cast(v_reg, tkl.f32), tkl.f16)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        casted = tkw.cast(res, datatype)
        tkw.write(
            casted, c, mapping=o_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    SHAPE = 0
    TILE_SIZE = 1

    symbols = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        B: batch[SHAPE],
        BN: n[SHAPE],
        K2: kv_seq_len[SHAPE],
        H: heads[SHAPE],
        K1: head_dim[SHAPE],
        M: q_seq_len[SHAPE],
        N: v_dim[SHAPE],
        BLOCK_B: batch[TILE_SIZE],
        BLOCK_BN: n[TILE_SIZE],
        BLOCK_H: heads[TILE_SIZE],
        BLOCK_M: q_seq_len[TILE_SIZE],
        BLOCK_N: v_dim[TILE_SIZE],
        BLOCK_K2: kv_seq_len[TILE_SIZE],
    }

    return evoformer_fwd, symbols
