# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    torch_dtype_to_wave,
)
from .attention_common import *
import math
import torch
import sympy
from typing import Optional


def get_extend_attention_kernel(
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.float32,
    size_dtype: torch.dtype = torch.int32,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    layer_scaling: Optional[float] = None,
    num_waves: int = 4,
    use_custom_mask: bool = False,
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
    assert not (
        is_causal and use_custom_mask
    ), f"Cannot have both causal and custom mask at the same time"

    S = tkl.sym.S
    EXT_IDX = tkl.sym.EXT_IDX
    KV_START_IDX = tkl.sym.KV_START_IDX
    MAX_EXTEND_SEQ_LEN = tkl.sym.MAX_EXTEND_SEQ_LEN
    MASK_LEN = tkl.sym.MASK_LEN
    MASK_START_IDX = tkl.sym.MASK_START_IDX
    SEQ_LEN = tkl.sym.SEQ_LEN
    PREFIX_LEN = tkl.sym.PREFIX_LEN
    # Workgroup tile sizes
    BLOCK_S = tkl.sym.BLOCK_S
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD_QK = index_symbol("LOAD_ELEMS_PER_THREAD_QK")
    LOAD_ELEMS_PER_THREAD_PV = index_symbol("LOAD_ELEMS_PER_THREAD_PV")
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    SEQ_TILE_SIZE = shape.block_size
    N_KV_SCALE = 2

    if SEQ_TILE_SIZE is None:
        # Apply tile heuristic.
        if shape.max_seq_len <= 128:
            SEQ_TILE_SIZE = 64
            N_KV_SCALE = 2
        else:
            SEQ_TILE_SIZE = 128
            N_KV_SCALE = 4

    M_WAVES = num_waves
    N_WAVES = 1
    LOG2E = 1.44269504089
    logit_cap *= LOG2E
    dk_sqrt = math.sqrt(1.0 / shape.head_size)
    layer_scaling = (layer_scaling or dk_sqrt) * LOG2E

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
    d0 = tkw.IndexMapping.dynamic_val(0)

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

    kv_indices_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={N_KV: i + KV_START_IDX},
        outputs={N_KV: i},
    )

    ind_ptr_mapping = tkw.IndexMapping(
        num_iterators=1,
        inputs={S: i + 1},
        outputs={S: i},
    )

    custom_mask_mapping_loop1 = tkw.IndexMapping(
        num_iterators=2,
        inputs={MASK_LEN: i * SEQ_LEN + MASK_START_IDX + j},
        outputs={N_Q: i, N_KV: j},
    )

    custom_mask_mapping_loop2 = tkw.IndexMapping(
        num_iterators=2,
        inputs={MASK_LEN: i * SEQ_LEN + MASK_START_IDX + j + PREFIX_LEN},
        outputs={N_Q: i, N_KV: j},
    )

    # Set the dynamic shapes for the kernel. Here we set it to N_Q
    # which is the first argument of the query and output.
    set_dynamic_dim = lambda shape: [x if i != 0 else None for i, x in enumerate(shape)]
    q_layout = tkl.MemoryLayout(shape=set_dynamic_dim(q_shape))
    k_layout = tkl.MemoryLayout(shape=set_dynamic_dim(k_shape))
    v_layout = tkl.MemoryLayout(shape=set_dynamic_dim(v_shape))
    o_layout = tkl.MemoryLayout(shape=set_dynamic_dim(o_shape))
    k_cache_layout = tkl.MemoryLayout(shape=k_cache_shape)
    v_cache_layout = tkl.MemoryLayout(shape=v_cache_shape)
    num_seqs_layout = tkl.MemoryLayout(shape=[None])
    kv_indices_layout = tkl.MemoryLayout(shape=[None])

    def extend_attention_core(
        q,
        k,
        v,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        c,
    ):
        c_reg = tkl.Register[H, D_KV, N_Q, tkl.f32](0.0)
        init_sum = tkl.Register[H, N_Q, tkl.f32](0.0)
        init_max = tkl.Register[H, N_Q, tkl.f32](-1e6)
        zero = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
        neg_infinity = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)
        layer_scale_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](layer_scaling)
        if logit_cap > 0:
            logit_cap_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](logit_cap)

        seq_extend_start_idx = tkw.read(qo_indptr, elements_per_thread=1)
        tkw.set_symbol(EXT_IDX, seq_extend_start_idx)
        seq_len_extend = (
            tkw.read(qo_indptr, elements_per_thread=1, mapping=ind_ptr_mapping)
            - seq_extend_start_idx
        )
        tkw.set_symbol(N_Q, seq_len_extend)
        seq_kv_start_idx = tkw.read(kv_indptr, elements_per_thread=1)
        tkw.set_symbol(KV_START_IDX, seq_kv_start_idx)
        seq_len_prefix = (
            tkw.read(kv_indptr, elements_per_thread=1, mapping=ind_ptr_mapping)
            - seq_kv_start_idx
        )
        tkw.set_symbol(N_KV, seq_len_prefix)
        if use_custom_mask:
            seq_len = seq_len_prefix + seq_len_extend
            tkw.set_symbol(SEQ_LEN, seq_len)
            seq_mask_start_idx = tkw.read(mask_offsets, elements_per_thread=1)
            tkw.set_symbol(MASK_START_IDX, seq_mask_start_idx)

        @tkw.iterate(N_KV, init_args=[init_max, init_sum, c_reg])
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
                kv_indices,
                elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
                mapping=kv_indices_mapping,
            )
            block_indices_k = tkw.read(
                kv_indices,
                elements_per_thread=1,
                mapping=kv_indices_mapping,
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
            if logit_cap > 0:
                logit_cap_reg_inv = tkw.reciprocal(logit_cap_reg)
                x_j = logit_cap_reg * tkw.tanh_approx(x_j * logit_cap_reg_inv)
            n_kv_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            mask = tkw.cast(mask, tkw.i1)
            if use_custom_mask:
                c_mask = tkw.read(
                    custom_mask,
                    elements_per_thread=STORE_ELEMS_PER_THREAD,
                    mapping=custom_mask_mapping_loop1,
                )
                c_mask = tkw.cast(c_mask, tkw.i1)
                mask &= c_mask
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

        if is_causal:
            seq_len_extend = tkw.apply_expr(
                seq_len_extend, lambda x: sympy.Min(x, (WORKGROUP_0 + 1) * BLOCK_N_Q)
            )
        tkw.set_symbol(N_KV, seq_len_extend)
        if use_custom_mask:
            tkw.set_symbol(PREFIX_LEN, seq_len_prefix)

        @tkw.iterate(N_KV, init_args=[res_max, res_sum, res_mm])
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
            x_j = x_j * layer_scale_reg
            if logit_cap > 0:
                logit_cap_reg_inv = tkw.reciprocal(logit_cap_reg)
                x_j = logit_cap_reg * tkw.tanh_approx(x_j * logit_cap_reg_inv)
            n_kv_index = tkw.self_index(N_KV, tkl.i32)
            mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
            mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
            if is_causal:
                n_q_index = tkw.self_index(N_Q, tkl.i32)
                n_q_index = tkw.broadcast(n_q_index, target_shape=[N_Q, N_KV])
                mask = (n_q_index >= n_kv_index) & mask
            mask = tkw.cast(mask, tkw.i1)
            if use_custom_mask:
                c_mask = tkw.read(
                    custom_mask,
                    elements_per_thread=STORE_ELEMS_PER_THREAD,
                    mapping=custom_mask_mapping_loop2,
                )
                c_mask = tkw.cast(c_mask, tkw.i1)
                mask &= c_mask
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

    @tkw.wave(constraints)
    def extend_attention(
        q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype, q_layout],
        k: tkl.Memory[N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_layout],
        v: tkl.Memory[N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_layout],
        k_cache: tkl.Memory[
            N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_cache_layout
        ],
        v_cache: tkl.Memory[
            N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_cache_layout
        ],
        qo_indptr: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout],
        kv_indptr: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout],
        kv_indices: tkl.Memory[N_KV, GLOBAL_ADDRESS_SPACE, tkl.i32, kv_indices_layout],
        c: tkl.Memory[N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype, o_layout],
    ):
        extend_attention_core(
            q, k, v, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, None, None, c
        )

    @tkw.wave(constraints)
    def extend_attention_custom_mask(
        q: tkl.Memory[N_Q, H, D_Q, GLOBAL_ADDRESS_SPACE, wave_input_dtype, q_layout],
        k: tkl.Memory[N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_layout],
        v: tkl.Memory[N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_layout],
        k_cache: tkl.Memory[
            N_KV, H_KV, D_Q, ADDRESS_SPACE, wave_input_dtype, k_cache_layout
        ],
        v_cache: tkl.Memory[
            N_KV, H_KV, D_KV, ADDRESS_SPACE, wave_input_dtype, v_cache_layout
        ],
        qo_indptr: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout],
        kv_indptr: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout],
        kv_indices: tkl.Memory[N_KV, GLOBAL_ADDRESS_SPACE, tkl.i32, kv_indices_layout],
        custom_mask: tkl.Memory[
            MASK_LEN, GLOBAL_ADDRESS_SPACE, tkl.i8, num_seqs_layout
        ],
        mask_offsets: tkl.Memory[S, GLOBAL_ADDRESS_SPACE, tkl.i32, num_seqs_layout],
        c: tkl.Memory[N_Q, H, D_KV, GLOBAL_ADDRESS_SPACE, wave_output_dtype, o_layout],
    ):
        extend_attention_core(
            q,
            k,
            v,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_offsets,
            c,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD_QK: get_mfma_load_elems_per_thread(mfma_variant[0]),
        LOAD_ELEMS_PER_THREAD_PV: get_mfma_load_elems_per_thread(mfma_variant[1]),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant[1]),
        BLOCK_H: 1,
        BLOCK_N_Q: SEQ_TILE_SIZE,
        BLOCK_D_KV: shape.head_size_kv,
        BLOCK_N_KV: SEQ_TILE_SIZE // N_KV_SCALE,
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

    if use_custom_mask:
        dynamic_symbols.append(MASK_LEN)
        dynamic_symbols_map[MASK_LEN] = shape.flattened_mask_len
        return (
            extend_attention_custom_mask,
            hyperparams,
            dynamic_symbols,
            dynamic_symbols_map,
        )

    return extend_attention, hyperparams, dynamic_symbols, dynamic_symbols_map
