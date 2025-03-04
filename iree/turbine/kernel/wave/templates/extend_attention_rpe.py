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
    torch_dtype_to_wave,
)
from .attention_common import *
import math
import torch
import sympy
from typing import Optional


def get_extend_attention_rpe_kernel(
    shape: AttentionShape,
    mfma_variant: MMAType,
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    block_table_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: Optional[torch.dtype] = torch.float16,
    output_dtype: Optional[torch.dtype] = torch.float32,
    size_dtype: Optional[torch.dtype] = torch.int32,
    is_causal: Optional[bool] = False,
    layer_scaling: Optional[float] = None,
    num_waves: Optional[int] = 4,
    max_rpe_context_length: Optional[int] = 0,
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
    EXT_IDX = tkl.sym.EXT_IDX
    REQ_IDX = tkl.sym.REQ_IDX
    SEQ_IDX = tkl.sym.SEQ_IDX
    MAX_EXTEND_SEQ_LEN = tkl.sym.MAX_EXTEND_SEQ_LEN
    # Workgroup tile sizes
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    SEQ_TILE_SIZE = shape.block_size
    M_WAVES = num_waves
    N_WAVES = 1
    LOG2E = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E
    rpe_scaling = LOG2E

    constraints: list[tkw.Constraint] = []
    constraints += [
        tkw.WorkgroupConstraint(
            N_Q, BLOCK_N_Q, 0, iters=sympy.ceiling(MAX_EXTEND_SEQ_LEN / SEQ_TILE_SIZE)
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
    d0, d1 = [tkw.IndexMapping.dynamic_val(i) for i in range(2)]

    mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, D_KV: j, N_Q: k},
        outputs={H: i, N_Q: k + EXT_IDX, D_KV: j},
    )

    q_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, N_Q: j + EXT_IDX, D_Q: k},
        outputs={H: i, N_Q: j, D_Q: k},
    )

    head_ratio = shape.num_query_heads // shape.num_kv_heads
    k_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, N_KV: j + EXT_IDX, D_Q: k},
        outputs={H_KV: i, N_KV: j, D_Q: k},
    )
    k_cache_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, N_KV: d0, D_Q: k},
        outputs={H_KV: i, N_KV: j, D_Q: k},
        dynamic_val_mappings={N_KV: j},
    )

    v_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, D_KV: j, N_KV: k + EXT_IDX},
        outputs={H_KV: i, D_KV: j, N_KV: k},
    )

    v_cache_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H_KV: i // head_ratio, D_KV: j, N_KV: d0},
        outputs={H_KV: i, D_KV: j, N_KV: k},
        dynamic_val_mappings={N_KV: k},
    )

    block_table_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={S: REQ_IDX, N_KV: i},
        outputs={N_KV: i},
    )

    clip = sympy.Piecewise(
        (d0 - d1, (d0 - d1 < max_rpe_context_length) & (d0 - d1 >= 0)),
        (max_rpe_context_length, True),
    )
    rpe_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={H: i, N_Q: j, N_KV: clip},
        outputs={H: i, N_Q: j, N_KV: k},
        dynamic_val_mappings=({H: i, N_Q: j, N_KV: k}, {N_KV: k}),
    )

    # Set the dynamic shapes for the kernel. Here we set it to N_Q
    # which is the first argument of the query and output.
    set_dynamic_dim = lambda shape: [x if i != 0 else None for i, x in enumerate(shape)]
    q_layout = tkl.MemoryLayout(shape=set_dynamic_dim(q_shape))
    k_layout = tkl.MemoryLayout(shape=set_dynamic_dim(k_shape))
    v_layout = tkl.MemoryLayout(shape=set_dynamic_dim(v_shape))
    o_layout = tkl.MemoryLayout(shape=set_dynamic_dim(o_shape))
    block_table_layout = tkl.MemoryLayout(shape=block_table_shape)
    k_cache_layout = tkl.MemoryLayout(shape=k_cache_shape)
    v_cache_layout = tkl.MemoryLayout(shape=v_cache_shape)
    num_seqs_layout = tkl.MemoryLayout(shape=[None])
    rpe_layout = tkl.MemoryLayout(shape=[max_rpe_context_length + 1])

    @tkw.wave(constraints)
    def extend_attention_rpe(
        q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype, q_layout],
        k: tkl.Memory[N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_layout],
        v: tkl.Memory[N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_layout],
        k_cache: tkl.Memory[
            N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_cache_layout
        ],
        v_cache: tkl.Memory[
            N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_cache_layout
        ],
        block_table: tkl.Memory[
            S, N_KV, GLOBAL_ADDRESS_SPACE, tkl.i32, block_table_layout
        ],
        request_indices: tkl.Memory[
            S, GLOBAL_ADDRESS_SPACE, wave_size_dtype, num_seqs_layout
        ],
        sequence_lengths: tkl.Memory[
            S, GLOBAL_ADDRESS_SPACE, wave_size_dtype, num_seqs_layout
        ],
        sequence_lengths_extend: tkl.Memory[
            S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout
        ],
        start_indices_extend: tkl.Memory[
            S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout
        ],
        rpe: tkl.Memory[N_KV, GLOBAL_ADDRESS_SPACE, tkl.f32, rpe_layout],
        c: tkl.Memory[N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype, o_layout],
    ):
        c_reg = tkl.Register[H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[H, N_Q, tkl.f32](-1e6)
        zero = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
        neg_infinity = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)
        layer_scale_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](layer_scaling)
        rpe_scale_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](rpe_scaling)

        req_idx = tkw.read(request_indices, elements_per_thread=1)
        tkw.set_symbol(REQ_IDX, req_idx)
        start = tkw.read(start_indices_extend, elements_per_thread=1)
        tkw.set_symbol(EXT_IDX, start)
        seq_len_extend = tkw.read(sequence_lengths_extend, elements_per_thread=1)
        tkw.set_symbol(N_Q, seq_len_extend)
        seq_len = tkw.read(sequence_lengths, elements_per_thread=1)
        seq_len = tkw.cast(seq_len, tkl.i32)
        seq_len_prefix = seq_len - seq_len_extend

        tkw.set_symbol(N_KV, seq_len_prefix)

        @tkw.reduction(N_KV, init_args=[init_max, init_sum, c_reg])
        def first_loop(
            partial_max: tkl.Register[H, N_Q, tkl.f32],
            partial_sum: tkl.Register[H, N_Q, tkl.f32],
            acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
        ):
            q_reg = tkw.read(
                q,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=q_mapping,
            )
            block_indices_v = tkw.read(
                block_table,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=block_table_mapping,
            )
            block_indices_k = tkw.read(
                block_table,
                elements_per_thread=1,
                mapping=block_table_mapping,
            )
            k_reg = tkw.read(
                k_cache,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
                mapping=k_cache_mapping,
                mapping_dynamic_vals=(block_indices_k,),
            )
            imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
            x_j = x_j * layer_scale_reg

            n_kv_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, zero, neg_infinity)
            x_j = x_j + bias

            m_j = tkw.max(x_j, partial_max, dim=N_KV)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=N_KV)
            imm_f16 = tkw.cast(e_delta, wave_input_dtype)
            v_reg = tkw.read(
                v_cache,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=v_cache_mapping,
                mapping_dynamic_vals=(block_indices_v,),
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = first_loop

        tkw.set_symbol(N_KV, seq_len_extend)

        @tkw.reduction(N_KV, init_args=[res_max, res_sum, res_mm])
        def second_loop(
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
            ####################################################################
            # T5 RPE
            ####################################################################
            # Fused T5 RPE adds attention bias pre-softmax normalization.
            # When fusing into the FA variant, adding locally before the max and
            # the partial softmax should be equivalent.
            i = tkw.self_index(N_Q, tkl.i64, elements_per_thread=1)
            i = tkw.broadcast(i, target_shape=[H, N_Q, N_KV])
            j = tkw.self_index(
                N_KV, tkl.i64, elements_per_thread=STORE_ELEMS_PER_THREAD
            )
            rpe_reg = tkw.read(
                rpe,
                mapping=rpe_mapping,
                mapping_dynamic_vals=(i, j),
                elements_per_thread=STORE_ELEMS_PER_THREAD,
            )
            # Layer and RPE scaling since we use log2 instead of ln
            x_j = x_j * layer_scale_reg + rpe_reg * rpe_scale_reg

            n_kv_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            if is_causal:
                n_q_index = tkw.self_index(N_Q, tkl.i32)
                n_q_index = tkw.broadcast(n_q_index, target_shape=[N_Q, N_KV])
                mask = (n_q_index >= n_kv_index) & mask
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, zero, neg_infinity)
            x_j = x_j + bias

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
        res_max, res_sum, res_mm = second_loop
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
        BLOCK_N_KV: SEQ_TILE_SIZE // 2,
        BLOCK_S: 1,
        H: shape.num_query_heads,
        H_KV: shape.num_kv_heads,
        D_KV: shape.head_size_kv,
        D_Q: shape.head_size,
    }

    dynamic_symbols = [N_Q, N_KV, S, MAX_EXTEND_SEQ_LEN]
    dynamic_symbols_map = {
        N_Q: q_shape[0],
        N_KV: k_shape[0],
        S: shape.num_seqs,
        MAX_EXTEND_SEQ_LEN: shape.max_seq_len,
    }

    return extend_attention_rpe, hyperparams, dynamic_symbols, dynamic_symbols_map
