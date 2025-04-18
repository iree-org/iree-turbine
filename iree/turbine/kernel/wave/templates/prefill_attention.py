# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
from typing import Optional

import iree.turbine.kernel.wave as tkw
import iree.turbine.kernel.lang as tkl
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
)
from iree.turbine.kernel.wave.templates.attention_common import *


def get_prefill_attention_kernel(
    shape: AttentionShape,
    mfma_variant: MMAType,
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: Optional[torch.dtype] = torch.float16,
    output_dtype: Optional[torch.dtype] = torch.float32,
    size_dtype: Optional[torch.dtype] = torch.int32,
):
    # Determine dtype of operands.
    wave_input_dtype = torch_dtype_to_wave(input_dtype)
    wave_output_dtype = torch_dtype_to_wave(output_dtype)
    wave_size_dtype = torch_dtype_to_wave(size_dtype)
    assert wave_input_dtype in [
        tkl.f16,
        tkl.bf16,
    ], f"Unsupported input datatype: {wave_input_dtype}"
    assert (
        wave_output_dtype.is_float_asm()
    ), f"Unsupported output datatype: {wave_output_dtype}"
    assert (
        wave_size_dtype.is_int_asm()
    ), f"Expected seq to be int but got: {wave_size_dtype}"

    S = tkl.sym.S
    OFFSET = tkl.sym.OFFSET
    # Workgroup tile sizes
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    SEQ_TILE_SIZE = 64
    M_WAVES = 4
    N_WAVES = 1

    constraints: list[tkw.Constraint] = []
    constraints += [
        tkw.WorkgroupConstraint(
            N_Q, BLOCK_N_Q, 0, iters=math.ceil(shape.max_seq_len / SEQ_TILE_SIZE)
        )
    ]
    constraints += [tkw.WorkgroupConstraint(D_KV, BLOCK_D_KV, 1)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 2)]
    constraints += [tkw.WorkgroupConstraint(H_KV, BLOCK_H, 2, primary=False)]
    constraints += [tkw.WorkgroupConstraint(S, BLOCK_S, 3)]
    constraints += [tkw.TilingConstraint(N_KV, BLOCK_N_KV)]
    constraints += [tkw.WaveConstraint(N_Q, BLOCK_N_Q / M_WAVES)]
    constraints += [tkw.WaveConstraint(D_KV, BLOCK_D_KV / N_WAVES)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(M_WAVES, N_WAVES, 1),
            mma_type=mfma_variant[1],
            vector_shapes={H: 0, H_KV: 0, N_Q: Mvec, D_KV: Nvec, S: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, D_KV: j, N_Q: k},
        outputs={H: i, N_Q: k + OFFSET, D_KV: j},
    )

    q_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, N_Q: j + OFFSET, D_Q: k},
        outputs={H: i, N_Q: j, D_Q: k},
    )

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    k_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, N_KV: j + OFFSET, D_Q: k},
        outputs={H_KV: i, N_KV: j, D_Q: k},
    )

    v_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, D_KV: j, N_KV: k + OFFSET},
        outputs={H_KV: i, D_KV: j, N_KV: k},
    )

    q_layout = tkl.MemoryLayout(shape=q_shape)
    k_layout = tkl.MemoryLayout(shape=k_shape)
    v_layout = tkl.MemoryLayout(shape=v_shape)
    o_layout = tkl.MemoryLayout(shape=o_shape)

    @tkw.wave(constraints)
    def prefill_attention(
        q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype, q_layout],
        k: tkl.Memory[N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_layout],
        v: tkl.Memory[N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_layout],
        offsets: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, wave_size_dtype],
        sequence_lengths: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, wave_size_dtype],
        c: tkl.Memory[N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype, o_layout],
    ):
        c_reg = tkl.Register[H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[H, N_Q, tkl.f32](-1e6)

        start = tkw.read(offsets, elements_per_thread=1)
        tkw.set_symbol(OFFSET, start)
        seq_len = tkw.read(sequence_lengths, elements_per_thread=1)
        tkw.set_symbol(N_Q, seq_len)
        tkw.set_symbol(N_KV, seq_len)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.iterate(N_KV, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[H, N_Q, tkl.f32],
            partial_sum: tkl.Register[H, N_Q, tkl.f32],
            acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
        ):
            imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
            q_reg = tkw.read(
                q,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=q_mapping,
            )
            k_reg = tkw.read(
                k,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=k_mapping,
            )
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, wave_input_dtype)
            v_reg = tkw.read(
                v,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=v_mapping,
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)
        res = res_mm * reciprocal_sum
        if wave_output_dtype != tkl.f32:
            res = tkw.cast(res, wave_output_dtype)
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK: get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV: get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_H: 1,
        BLOCK_N_Q: SEQ_TILE_SIZE,
        BLOCK_D_KV: SEQ_TILE_SIZE,
        BLOCK_N_KV: SEQ_TILE_SIZE,
        BLOCK_S: 1,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
        S: shape.num_seqs,
    }

    return prefill_attention, hyperparams
