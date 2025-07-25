# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.nn import functional as F
import math
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType
from ..common.utils import (
    dump_generated_mlir,
    enable_scheduling_barriers,
    expensive_test,
    require_e2e,
)
from torch.testing import assert_close

# batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len
shapes_16x16x16 = [
    # Test first with very small shapes. These are much easier to debug.
    (1, 16, 16, 16, 16),
    (1, 16, 16, 16, 32),
    (1, 16, 16, 32, 16),
    (1, 16, 16, 64, 16),
    (1, 16, 32, 16, 16),
    (1, 32, 16, 16, 16),
    (2, 16, 16, 16, 16),
]

shapes_32x32x32 = [
    tuple(dim if i == 0 else 2 * dim for i, dim in enumerate(shape))
    for shape in shapes_16x16x16
]


big_shapes = [
    # Bigger shapes
    (2, 64, 128, 32, 256),
    (2, 1024, 64, 64, 1024),
    (8, 128, 128, 64, 256),
    # The batch size 40 is used in other attention tests, but mostly just makes
    # things slower. I don't think it helps that much with correctness testing
    # and we're not doing performance testing yet.
    # (40, 1024, 64, 64, 1024),
]


def get_param_id(val):
    if isinstance(val, tuple) and all(isinstance(el, int) for el in val):
        return "x".join(str(el) for el in val)
    elif isinstance(val, MMAType):
        return f"MMA_{val.name}"


param_mfma_shape = pytest.mark.parametrize(
    "mfma_variant,shape",
    [(MMAType.F32_16x16x16_F16, shape) for shape in shapes_16x16x16 + big_shapes]
    + [(MMAType.F32_32x32x8_F16, shape) for shape in shapes_32x32x32 + big_shapes],
    ids=get_param_id,
)


def attention_torch_builtin_ref(q, k, v, do, scale=1):
    """Attention forward and backward reference using the Torch builtin."""
    q = q.clone().requires_grad_()
    v = v.clone().requires_grad_()
    k = k.clone().requires_grad_()

    o = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        scale=scale,
    )

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o.backward(do)

    dq = q.grad.detach()
    dk = k.grad.detach()
    dv = v.grad.detach()
    return o, dq, dk, dv


def attention_torch_ops_ref(q, k, v, do, scale=1):
    """Attention reference computed with individual Torch operations.

    This gives us a more consistent reference since the Torch builtin uses
    different implementations depending on the platform and other factors. It
    also gives us reference values for the intermediates of the computation,
    which enables more precise testing and debugging of our kernels.
    """
    q = q.clone().requires_grad_()
    v = v.clone().requires_grad_()
    k = k.clone().requires_grad_()

    s = scale * torch.matmul(q, k.transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)

    m = torch.max(s, dim=-1)[0]
    p_prime = torch.exp(s - m.unsqueeze(-1))
    lse = m + torch.log(torch.sum(p_prime, dim=-1))

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    s.retain_grad()
    p.retain_grad()

    o.backward(do)
    dq = q.grad.detach()
    dk = k.grad.detach()
    dv = v.grad.detach()
    ds = s.grad.detach()
    dp = p.grad.detach()
    o = o.detach()
    s = s.detach()
    p = p.detach()

    return o, lse, dq, dk, dv, s, p, ds, dp


def get_attention_fwd_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
    scale: float,
):
    """Flash Attention 2 forward kernel.

    Includes outputting intermediate values so that they can be verified for
    testing, which obviously we would not want to do when optimizing for
    performance.
    """
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M_r_qs = tkl.sym.BLOCK_M
    BLOCK_N_vd = tkl.sym.BLOCK_N
    BLOCK_K2_c_kvs = tkl.sym.BLOCK_K2
    # Set these all to 1 for now. It's also not clear why these can't be symbols
    # in the hyperparams.
    WAVES_PER_BLOCK_M = 1
    WAVES_PER_BLOCK_N = 1
    WAVES_PER_BLOCK_B = 1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M_qs, BLOCK_M_r_qs, 0)]
    constraints += [tkw.WorkgroupConstraint(N_vd, BLOCK_N_vd, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2_kvs, BLOCK_K2_c_kvs)]
    constraints += [tkw.WaveConstraint(M_qs, BLOCK_M_r_qs / WAVES_PER_BLOCK_M)]
    constraints += [tkw.WaveConstraint(N_vd, BLOCK_N_vd / WAVES_PER_BLOCK_N)]
    constraints += [tkw.WaveConstraint(B, BLOCK_B / WAVES_PER_BLOCK_B)]

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    if mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_n_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, N_vd: j, M_qs: k},
        outputs={B: i, M_qs: k, N_vd: j},
    )

    # TODO: tune address space
    @tkw.wave(constraints)
    def attention_fwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N_vd, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # We only output s so that we can verify it in the test. Obviously,
        # doing so defeats the entire purpose of Flash Attention from a
        # performance perspective.
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        o: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        init_m = tkl.Register[B, M_qs, tkl.f32](-1e6)
        init_l = tkl.Register[B, M_qs, tkl.f32](0.0)
        init_o = tkl.Register[B, N_vd, M_qs, tkl.f32](0.0)

        @tkw.iterate(K2_kvs, init_args=[init_m, init_l, init_o])
        def repeat(
            m_prev: tkl.Register[B, M_qs, tkl.f32],
            l_prev: tkl.Register[B, M_qs, tkl.f32],
            o_prev: tkl.Register[B, N_vd, M_qs, tkl.f32],
        ):
            s_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
            scale_reg = tkl.Register[B, K2_kvs, M_qs, tkl.f32](scale)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            s_unscaled_ij = tkw.mma(k_j, q_i, s_acc)
            # TODO(#410): This no-op permute works around expansion failing in
            # the K1 dimension when the scaling factor is applied.
            s_unscaled_ij = tkw.permute(s_unscaled_ij, [B, K2_kvs, M_qs])
            s_ij = scale_reg * s_unscaled_ij
            s_ij = tkw.permute(s_ij, target_shape=[B, M_qs, K2_kvs])
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            m_ij = tkw.max(s_ij, m_prev, dim=K2_kvs)
            e_delta_max = tkw.exp2(log2e * (m_prev - m_ij))
            p_ij = tkw.exp2(log2e * (s_ij - m_ij))
            l_init = e_delta_max * l_prev
            l_ij = tkw.sum(p_ij, l_init, dim=K2_kvs)
            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            o_init = e_delta_max * o_prev
            o_ij = tkw.mma(v_j, tkw.cast(p_ij, tkl.f16), o_init)
            return m_ij, l_ij, o_ij

        log2e = tkl.Register[B, M_qs, tkl.f32](1.44269504089)
        m_i, l_i, o_i = repeat
        o_i = tkw.reciprocal(l_i) * o_i
        tkw.write(
            tkw.cast(o_i, tkl.f16),
            o,
            mapping=flip_n_m_write_mapping,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )
        lse_i = m_i + (tkw.log2(l_i) / log2e)
        tkw.write(tkw.cast(lse_i, tkl.f16), lse, elements_per_thread=1)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M_r_qs: vec_size,
        BLOCK_N_vd: vec_size,
        BLOCK_K2_c_kvs: vec_size,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_fwd, hyperparams


def get_attention_bwd_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
    scale: float,
):
    """Flash Attention 2 backward kernel.

    Includes outputting intermediate values so that they can be verified for
    testing, which obviously we would not want to do when optimizing for
    performance.

    This also currently has to perform a lot of extra reads to satisfy the
    layout restrictions wave has for MMAs, including having to write out and
    then read back an intermediate variable. Hopefully these can be elimanted.
    """
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # TODO(#392): Can only have 3 dimensions distributed in actual blocks or
        # the compiler tries to index too far into waves_per_block (and if that
        # is made longer there's just a fatal crash), so batch dimension needs
        # to be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            # TODO(#384): If we don't set the N vector shape here, then there is
            # a compilation failure when BLOCK_N (which is just set to the head
            # dimension) is not equal to vec_size.
            vector_shapes={B: 0, N_vd: vec_size},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )
    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, M_qs: k},
    )
    flip_m_k2_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K2_kvs: j},
        outputs={B: i, K2_kvs: j, M_qs: k},
    )
    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, K2_kvs: k},
    )

    @tkw.wave(constraints)
    def attention_bwd(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # We have extra output arguments so we can check intermediates. Obiously
        # doing this is not at all performant.
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds_scaled: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)
        dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        @tkw.iterate(M_qs, init_args=[dv_init, dk_init])
        def loop_q_seq_len(
            dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32],
            dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32],
        ):
            # TODO(#602): Wave has implicit layout requirements for MMAs, so we
            # have to actually compute s transpose and then permute it.
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            scale_s_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f32](scale)
            s_unscaled_ij = tkw.mma(q_i, k_j, s_acc)
            # TODO(#410): This no-op permute works around either index
            # assignment or expansion failing when doing a subsequent
            # elementwise operation.
            s_unscaled_ij = tkw.permute(s_unscaled_ij, [B, M_qs, K2_kvs])
            s_ij = scale_s_reg * s_unscaled_ij
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            s_ij = tkw.permute(s_ij, [B, K2_kvs, M_qs])
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(
                p_ij,
                p,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
            )

            do_i_for_dv = tkw.read(
                do,
                mapping=flip_m_n_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dv_j = tkw.mma(p_ij, do_i_for_dv, dv_prev)

            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have load do_i again in this layout.
            do_i_for_dp = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            dp_ij = tkw.mma(v_j, do_i_for_dp, dp_acc)
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=1)
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - tkw.broadcast(D_i, [B, M_qs, K2_kvs])
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            # Just multiplying p_ij * dp_ij_sub breaks the previously calculated
            # dp. We have to load back p in the required layout.
            scale_ds_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f16](scale)
            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read back p_ij in this layout.
            p_ij_for_ds = tkw.read(p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            ds_ij = p_ij_for_ds * dp_ij_sub
            tkw.write(ds_ij, ds, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            ds_scaled_ij = scale_ds_reg * ds_ij
            tkw.write(
                ds_scaled_ij, ds_scaled, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD
            )

            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read back ds_ij in this layout.
            ds_scaled_ij_for_dk = tkw.read(
                ds_scaled,
                mapping=flip_m_k2_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read q again.
            q_i_for_dk = tkw.read(
                q,
                mapping=flip_m_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dk_j = tkw.mma(ds_scaled_ij_for_dk, q_i_for_dk, dk_prev)

            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read k again.
            k_j_for_dq = tkw.read(
                k,
                mapping=flip_k2_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dq_prev = tkw.read(dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dq_i = tkw.mma(ds_scaled_ij, k_j_for_dq, tkw.cast(dq_prev, tkl.f32))
            tkw.write(dq_i, dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            return (
                dv_j,
                dk_j,
            )

        dv_j, dk_j = loop_q_seq_len
        tkw.write(
            tkw.cast(dv_j, tkl.f16), dv, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD
        )
        tkw.write(
            tkw.cast(dk_j, tkl.f16), dk, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # TODO(#364) and TODO(#365) and TODO(#586): We actually want a nested
        # (#586) loop (#364) or an atomic add (#365), but those aren't
        # supported. So we force the distribution of K2 to be degenerate and
        # expansion will have to fully unroll it.
        BLOCK_K2: max(kv_seq_len, vec_size),
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd, hyperparams


def get_attention_bwd_dv_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
    scale: float,
):
    """Flash Attention 2 backward kernel for dv only.

    Isolates the dv calculation so that we can work on working through bugs and
    extra reads separately.
    """
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # TODO(#392): Can only have 3 dimensions distributed in actual blocks or
        # the compiler tries to index too far into waves_per_block (and if that
        # is made longer there's just a fatal crash), so batch dimension needs
        # to be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)

    flip_m_n_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, N_vd: j},
        outputs={B: i, N_vd: j, M_qs: k},
    )

    # Other than writing out the intermediate values (which is just a debugging
    # aid), this kernel is basically fine with respect to extra reads or writes
    # of the same tensor. It still has to have fake tiling of dimensions that
    # aren't actually being tiled.
    @tkw.wave(constraints)
    def attention_bwd_dv(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dv: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dv_init = tkl.Register[B, K2_kvs, N_vd, tkl.f32](0.0)

        @tkw.iterate(M_qs, init_args=[dv_init])
        def loop_q_seq_len(dv_prev: tkl.Register[B, K2_kvs, N_vd, tkl.f32]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            scale_s_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f32](scale)
            s_unscaled_ij = tkw.mma(q_i, k_j, s_acc)
            # TODO(#410): This no-op permute works around either index
            # assignment or expansion failing when doing a subsequent
            # elementwise operation. I think it just hides the K1 dimension from
            # the index.
            s_unscaled_ij = tkw.permute(s_unscaled_ij, [B, M_qs, K2_kvs])
            s_ij = scale_s_reg * s_unscaled_ij
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.permute(p_ij, [B, K2_kvs, M_qs])

            do_i = tkw.read(
                do,
                mapping=flip_m_n_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dv_j = tkw.mma(p_ij, do_i, dv_prev)
            return dv_j

        tkw.write(
            tkw.cast(loop_q_seq_len, tkl.f16),
            dv,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # distributing K2 speeds things up a ton. We can do this because we're
        # not computing dq.
        BLOCK_K2: vec_size,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd_dv, hyperparams


def get_attention_bwd_dk_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
    scale: float,
):
    """Flash Attention 2 backward kernel for dk only.

    Isolates the dk calculation so that we can work on working through bugs and
    extra reads separately.
    """
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.MFMA_INPUT_ELS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.MFMA_OUTPUT_ELS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # TODO(#392): Can only have 3 dimensions distributed in actual blocks or
        # the compiler tries to index too far into waves_per_block (and if that
        # is made longer there's just a fatal crash), so batch dimension needs
        # to be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_m_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, M_qs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, M_qs: k},
    )

    # We're able to eliminate some of the issues with the combined kernel, like
    # having to pre-broadcast lse and write out and read back ds, but this still
    # has an extra read of q.
    @tkw.wave(constraints)
    def attention_bwd_dk(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dk: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # We have extra output arguments so we can check intermediates. Obiously
        # doing this is not at all performant.
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        dk_init = tkl.Register[B, K2_kvs, K1_qkd, tkl.f32](0.0)

        @tkw.iterate(M_qs, init_args=[dk_init])
        def loop_q_seq_len(dk_prev: tkl.Register[B, K2_kvs, K1_qkd, tkl.f32]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            log2e = tkl.Register[B, M_qs, K2_kvs, tkl.f16](1.44269504089)
            scale_s_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f32](scale)
            s_unscaled_ij = tkw.mma(q_i, k_j, s_acc)
            # TODO(#410): This no-op permute works around either index
            # assignment or expansion failing when doing a subsequent
            # elementwise operation. I think it just hides the K1 dimension from
            # the index.
            s_unscaled_ij = tkw.permute(s_unscaled_ij, [B, M_qs, K2_kvs])
            s_ij = scale_s_reg * s_unscaled_ij
            tkw.write(s_ij, s, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            lse_i = tkw.read(lse, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.exp2(log2e * (tkw.cast(s_ij, tkl.f16) - lse_i))
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            p_ij = tkw.permute(p_ij, [B, K2_kvs, M_qs])

            dp_acc = tkl.Register[B, M_qs, K2_kvs, tkl.f32](0.0)
            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            do_i = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_ij = tkw.mma(do_i, v_j, dp_acc)
            # TODO(#410): this no-op permute fixes a compiler error by hiding
            # the N index of the mma from the cast that uses it. Otherwise, the
            # cast operation fails to update the op it uses during expansion.
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            tkw.write(dp_ij, dp, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            D_i = tkw.read(D, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - D_i
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dp_ij_sub = tkw.permute(dp_ij_sub, [B, K2_kvs, M_qs])

            ds_ij = p_ij * dp_ij_sub
            tkw.write(
                ds_ij,
                ds,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
            )
            scale_ds_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f16](scale)
            ds_scaled_ij = scale_ds_reg * ds_ij

            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read q again.
            q_i_for_dk = tkw.read(
                q,
                mapping=flip_m_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dk_j = tkw.mma(ds_scaled_ij, q_i_for_dk, dk_prev)

            return dk_j

        tkw.write(
            tkw.cast(loop_q_seq_len, tkl.f16),
            dk,
            elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # distributing K2 speeds things up a ton. We can do this because we're
        # not computing dq.
        BLOCK_K2: vec_size,
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd_dk, hyperparams


def get_attention_bwd_dq_kernel(
    batch: int,
    kv_seq_len: int,
    qk_head_dim: int,
    q_seq_len: int,
    v_head_dim: int,
    mfma_variant: MMAType,
    scale: float,
):
    """Flash Attention 2 backward kernel for dq only.

    Isolates the dq calculation so that we can work on working through bugs and
    extra reads separately.
    """
    # Input sizes
    B = tkl.sym.B  # batch
    M_qs = tkl.sym.M  # query sequence length
    N_vd = tkl.sym.N  # value head dimension
    K1_qkd = tkl.sym.K1  # query/key head dimension
    K2_kvs = tkl.sym.K2  # key/value sequence length

    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K1 = tkl.sym.BLOCK_K1
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    MFMA_INPUT_ELS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    MFMA_OUTPUT_ELS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    if mfma_variant == MMAType.F32_16x16x16_F16:
        vec_size = 16
    elif mfma_variant == MMAType.F32_32x32x8_F16:
        vec_size = 32

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [
        tkw.WorkgroupConstraint(K2_kvs, BLOCK_K2, 0),
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        tkw.WorkgroupConstraint(K1_qkd, BLOCK_K1, 1),
        tkw.WorkgroupConstraint(N_vd, BLOCK_N, 2),
        # TODO(#392): Can only have 3 dimensions distributed in actual blocks or
        # the compiler tries to index too far into waves_per_block (and if that
        # is made longer there's just a fatal crash), so batch dimension needs
        # to be last.
        tkw.WorkgroupConstraint(B, BLOCK_B, 3),
        tkw.TilingConstraint(M_qs, BLOCK_M),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    flip_k2_m_write_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: j, M_qs: k},
        outputs={B: i, M_qs: k, K2_kvs: j},
    )

    flip_k2_k1_read_mapping = tkw.IndexMapping(
        num_iterators=3,
        inputs={B: i, K2_kvs: k, K1_qkd: j},
        outputs={B: i, K1_qkd: j, K2_kvs: k},
    )

    # We're able to eliminate some of the issues with the combined kernel, like
    # having to pre-broadcast lse and write out and read back ds, but this still
    # has an extra read of k, and has to do a full expansion (i.e. unrolled loop
    # in the K2 dimension).
    @tkw.wave(constraints)
    def attention_bwd_dq(
        q: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2_kvs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, K2_kvs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        do: tkl.Memory[B, M_qs, N_vd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        lse: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        D: tkl.Memory[B, M_qs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dq: tkl.Memory[B, M_qs, K1_qkd, GLOBAL_ADDRESS_SPACE, tkl.f16],
        # We have extra output arguments so we can check intermediates. Obiously
        # doing this is not at all performant.
        s: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        s_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        p: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        ds: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
        dp: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f32],
        dp_sub: tkl.Memory[B, M_qs, K2_kvs, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):

        # TODO(#364): Workaround for missing non-iterate loop. This needs to
        # have only dimensions that are in the vector shapes or it doesn't have
        # vector shapes for its indexing dims.
        dummy_init = tkl.Register[B, tkl.f16](0.0)

        @tkw.iterate(M_qs, init_args=[dummy_init])
        def loop_q_seq_len(dummy_prev: tkl.Register[B, tkl.f16]):
            k_j = tkw.read(k, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            q_i = tkw.read(q, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)

            s_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            scale_s_reg = tkl.Register[B, K2_kvs, M_qs, tkl.f32](scale)
            s_unscaled_ij = tkw.mma(k_j, q_i, s_acc)
            s_unscaled_ij = tkw.permute(s_unscaled_ij, [B, K2_kvs, M_qs])
            s_ij = scale_s_reg * s_unscaled_ij
            # permuting and then writing without a mapping breaks whichever of s
            # and dp is used later in the kernel iff we multiply p_ij and
            # dp_ij_sub to compute ds_ij. So even though we are about to permute
            # this, we do the write with a mapping instead.
            tkw.write(
                s_ij,
                s,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
            )
            s_ij = tkw.permute(s_ij, [B, M_qs, K2_kvs])
            # For some reason because s is now computed as K2xM and then
            # permuted to MxK2 (as opposed to the reverse in the fused kernel),
            # elements_per_thread needs to be 1 instead of
            # MFMA_OUTPUT_ELS_PER_THREAD. I'm not sure which is more desirable.
            lse_i = tkw.read(lse, elements_per_thread=1)
            s_ij_sub = tkw.cast(s_ij, tkl.f16) - lse_i
            tkw.write(s_ij_sub, s_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            log2e = tkl.Register[B, K2_kvs, M_qs, tkl.f16](1.44269504089)
            log2e = tkw.permute(log2e, [B, M_qs, K2_kvs])
            p_ij = tkw.exp2(log2e * s_ij_sub)
            tkw.write(p_ij, p, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            v_j = tkw.read(v, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            do_i = tkw.read(do, elements_per_thread=MFMA_INPUT_ELS_PER_THREAD)
            dp_acc = tkl.Register[B, K2_kvs, M_qs, tkl.f32](0.0)
            dp_ij = tkw.mma(v_j, do_i, dp_acc)
            # permuting and then writing without a mapping breaks whichever of s
            # and dp is used later in the kernel iff we multiply p_ij and
            # dp_ij_sub to compute ds_ij.
            tkw.write(
                dp_ij,
                dp,
                mapping=flip_k2_m_write_mapping,
                elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD,
            )
            dp_ij = tkw.permute(dp_ij, [B, M_qs, K2_kvs])
            D_i = tkw.read(D, elements_per_thread=1)
            dp_ij_sub = tkw.cast(dp_ij, tkl.f16) - D_i
            tkw.write(dp_ij_sub, dp_sub, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            ds_ij = p_ij * dp_ij_sub
            tkw.write(ds_ij, ds, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            scale_ds_reg = tkl.Register[B, M_qs, K2_kvs, tkl.f16](scale)
            ds_scaled_ij = scale_ds_reg * ds_ij

            # TODO(#603): Wave has implicit layout requirements for MMAs, so we
            # have to read k again.
            k_j_for_dq = tkw.read(
                k,
                mapping=flip_k2_k1_read_mapping,
                elements_per_thread=MFMA_INPUT_ELS_PER_THREAD,
            )
            dq_prev = tkw.read(dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)
            dq_i = tkw.mma(ds_scaled_ij, k_j_for_dq, tkw.cast(dq_prev, tkl.f32))
            tkw.write(dq_i, dq, elements_per_thread=MFMA_OUTPUT_ELS_PER_THREAD)

            return dummy_prev

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        MFMA_INPUT_ELS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        MFMA_OUTPUT_ELS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: vec_size,
        # TODO(#381) and TODO(#384): We need the degenerate tiling to make Wave
        # recognize the dims, but it needs to be at least the vector size or we
        # get errors about tile size being divisible by vector size.
        BLOCK_N: max(v_head_dim, vec_size),
        BLOCK_K1: max(qk_head_dim, vec_size),
        # TODO(#364) and TODO(#365) and TODO(#586): We actually want a nested
        # (#586) loop (#364) or an atomic add (#365), but those aren't
        # supported. So we force the distribution of K2 to be degenerate and
        # expansion will have to fully unroll it.
        BLOCK_K2: max(kv_seq_len, vec_size),
        B: batch,
        M_qs: q_seq_len,
        N_vd: v_head_dim,
        K1_qkd: qk_head_dim,
        K2_kvs: kv_seq_len,
    }

    return attention_bwd_dq, hyperparams


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@pytest.mark.parametrize("shape", big_shapes, ids=get_param_id)
def testAttentionOpsReference(shape: tuple[int, ...]):
    """Validate our manual reference implementation against the torch builtin.

    We subsequently use it as a reference for the Wave kernels because it's more
    stable and includes intermediate values we can test against.
    """

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape

    scale = math.sqrt(1.0 / qk_head_dim)

    q = device_randn(batch, q_seq_len, qk_head_dim)
    k = device_randn(batch, kv_seq_len, qk_head_dim)
    v = device_randn(batch, kv_seq_len, v_head_dim)
    do = device_randn(batch, q_seq_len, v_head_dim)

    o_ref, dq_ref, dk_ref, dv_ref = attention_torch_builtin_ref(
        q, k, v, do, scale=scale
    )

    (
        o_ops,
        unused_lse_ops,
        dq_ops,
        dk_ops,
        dv_ops,
        unused_s_ops,
        unused_p_ops,
        unused_ds_ops,
        unused_dp_ops,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    assert_close(o_ops, o_ref)
    assert_close(dq_ops, dq_ref)
    assert_close(dk_ops, dk_ref)
    assert_close(dv_ops, dv_ref)


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@param_mfma_shape
def testAttentionForward(mfma_variant: MMAType, shape: tuple[int, ...]):
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    scale = math.sqrt(1.0 / qk_head_dim)
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

    # We could move the reference to a fixture, but it's pretty fast, so that
    # doesn't seem worth the extra complexity.
    (
        o_ref,
        lse_ref,
        unused_dq_ref,
        unused_dk_ref,
        unused_dv_ref,
        s_ref,
        p_ref,
        unused_ds_ref,
        unused_dp_ref,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    attention_fwd, hyperparams = get_attention_fwd_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
        scale=scale,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    attention_fwd = wave_compile(options, attention_fwd)

    o = device_zeros(batch, q_seq_len, v_head_dim, dtype=torch.float16)
    lse = device_zeros(batch, q_seq_len, dtype=torch.float16)
    s = device_zeros(batch, q_seq_len, kv_seq_len)

    asm_fwd = attention_fwd(q, k, v.transpose(-1, -2), s, o, lse)

    if dump_generated_mlir:
        filename = f"out/wave_attention_fwd_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_fwd)
        print(f"IR dumped to {filename}")

    assert_close(s, s_ref, **cmp_params)
    # Can't check P, since we don't actually compute the "real" thing in the
    # forward pass, but rather rescale as we go.
    assert_close(lse, lse_ref, **cmp_params)
    assert_close(o, o_ref, **cmp_params)


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@param_mfma_shape
@expensive_test
def testAttentionBackward(mfma_variant: MMAType, shape: tuple[int, ...]):
    if mfma_variant == MMAType.F32_32x32x8_F16:
        pytest.skip("Asymmetric MFMA is broken")

    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    scale = math.sqrt(1.0 / qk_head_dim)
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

    (
        o_ref,
        lse_ref,
        dq_ref,
        dk_ref,
        dv_ref,
        s_ref,
        p_ref,
        ds_ref,
        dp_ref,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    attention_bwd, hyperparams = get_attention_bwd_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
        scale=scale,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    attention_bwd = wave_compile(options, attention_bwd)

    D = torch.sum(do * o_ref, -1)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
    p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
    ds = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
    ds_scaled = torch.zeros_like(ds)
    dp = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
    dp_sub = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

    asm_bwd = attention_bwd(
        q,
        k,
        v,
        do,
        lse_ref,
        D,
        dq,
        dk,
        dv,
        s,
        p,
        ds,
        ds_scaled,
        dp,
        dp_sub,
    )

    if dump_generated_mlir:
        filename = f"out/wave_attention_bwd_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_bwd)
        print(f"IR dumped to {filename}")

    assert_close(s, s_ref, **cmp_params)
    assert_close(p, p_ref, **cmp_params)

    assert_close(dv, dv_ref, **cmp_params)

    dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)
    assert_close(dp, dp_ref, **cmp_params)
    assert_close(dp_sub, dp_sub_ref, **cmp_params)

    assert_close(ds, ds_ref, **cmp_params)

    assert_close(dk, dk_ref, **cmp_params)
    assert_close(dq, dq_ref, **cmp_params)


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@param_mfma_shape
def testAttentionBackward_dv(mfma_variant: MMAType, shape: tuple[int, ...]):
    """This tests a kernel only for the gradient of v."""
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    scale = math.sqrt(1.0 / qk_head_dim)
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

    (
        unused_o_ref,
        lse_ref,
        unused_dq_ref,
        unused_dk_ref,
        dv_ref,
        s_ref,
        p_ref,
        unused_ds_ref,
        unused_dp_ref,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    attention_bwd_dv, hyperparams_dv = get_attention_bwd_dv_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
        scale=scale,
    )
    hyperparams_dv.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams_dv,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    attention_bwd_dv = wave_compile(options, attention_bwd_dv)

    dv = torch.zeros_like(v)
    s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
    p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)

    asm_bwd_dv = attention_bwd_dv(q, k, do, lse_ref, dv, s, p)

    if dump_generated_mlir:
        filename = f"out/wave_attention_bwd_dv_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_bwd_dv)
        print(f"IR dumped to {filename}")

    assert_close(s, s_ref, **cmp_params)
    assert_close(p, p_ref, **cmp_params)
    assert_close(dv, dv_ref, **cmp_params)


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@param_mfma_shape
def testAttentionBackward_dk(mfma_variant: MMAType, shape: tuple[int, ...]):
    """This tests a kernel only for the gradient of k."""
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    scale = math.sqrt(1.0 / qk_head_dim)
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

    (
        o_ref,
        lse_ref,
        unused_dq_ref,
        dk_ref,
        unused_dv_ref,
        s_ref,
        p_ref,
        ds_ref,
        dp_ref,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    attention_bwd_dk, hyperparams_dk = get_attention_bwd_dk_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
        scale=scale,
    )
    hyperparams_dk.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams_dk,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    attention_bwd_dk = wave_compile(options, attention_bwd_dk)

    D = torch.sum(do * o_ref, -1).to(torch.float16)
    dk = torch.zeros_like(k)
    s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
    p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
    ds = torch.zeros_like(p)
    dp = torch.zeros_like(s)
    dp_sub = torch.zeros_like(p)

    asm_bwd_dk = attention_bwd_dk(
        q,
        k,
        v,
        do,
        lse_ref,
        D,
        dk,
        s,
        p,
        ds,
        dp,
        dp_sub,
    )

    if dump_generated_mlir:
        filename = f"out/wave_attention_bwd_dk_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_bwd_dk)
        print(f"IR dumped to {filename}")

    dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)

    assert_close(s, s_ref, **cmp_params)
    assert_close(p, p_ref, **cmp_params)
    assert_close(dp, dp_ref, **cmp_params)
    assert_close(dp_sub, dp_sub_ref, **cmp_params)
    assert_close(ds, ds_ref, **cmp_params)
    assert_close(dk, dk_ref, **cmp_params)


@pytest.mark.skip(reason="Flaky + Wave is moving out of the repo")
@require_e2e
@param_mfma_shape
@expensive_test
def testAttentionBackward_dq(mfma_variant: MMAType, shape: tuple[int, ...]):
    """This tests a kernel only for the gradient of q."""
    batch, q_seq_len, v_head_dim, qk_head_dim, kv_seq_len = shape
    scale = math.sqrt(1.0 / qk_head_dim)
    cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)

    q = device_randn(batch, q_seq_len, qk_head_dim, dtype=torch.float16)
    k = device_randn(batch, kv_seq_len, qk_head_dim, dtype=torch.float16)
    v = device_randn(batch, kv_seq_len, v_head_dim, dtype=torch.float16)
    do = device_randn(batch, q_seq_len, v_head_dim, dtype=torch.float16)

    (
        o_ref,
        lse_ref,
        dq_ref,
        unused_dk_ref,
        unused_dv_ref,
        s_ref,
        p_ref,
        ds_ref,
        dp_ref,
    ) = attention_torch_ops_ref(q, k, v, do, scale=scale)

    attention_bwd_dq, hyperparams_dq = get_attention_bwd_dq_kernel(
        batch=batch,
        kv_seq_len=kv_seq_len,
        qk_head_dim=qk_head_dim,
        q_seq_len=q_seq_len,
        v_head_dim=v_head_dim,
        mfma_variant=mfma_variant,
        scale=scale,
    )
    hyperparams_dq.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams_dq,
        use_scheduling_barriers=enable_scheduling_barriers,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    attention_bwd_dq = wave_compile(options, attention_bwd_dq)

    D = torch.sum(do * o_ref, -1).to(torch.float16)
    dq = torch.zeros_like(q)
    s = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float32)
    p = device_zeros(batch, q_seq_len, kv_seq_len, dtype=torch.float16)
    s_sub = torch.zeros_like(p)
    ds = torch.zeros_like(p)
    dp = torch.zeros_like(s)
    dp_sub = torch.zeros_like(p)

    asm_bwd_dq = attention_bwd_dq(
        q,
        k,
        v,
        do,
        lse_ref,
        D,
        dq,
        s,
        s_sub,
        p,
        ds,
        dp,
        dp_sub,
    )

    if dump_generated_mlir:
        filename = f"out/wave_attention_bwd_dq_{'x'.join(map(str, shape))}.mlir"
        with open(filename, "w") as f:
            f.write(asm_bwd_dq)
        print(f"IR dumped to {filename}")

    s_sub_ref = s_ref.to(torch.float16) - lse_ref.reshape((batch, q_seq_len, 1)).expand(
        batch, q_seq_len, kv_seq_len
    )
    dp_sub_ref = (dp_ref - D.reshape((batch, q_seq_len, 1))).to(torch.float16)

    assert_close(s, s_ref, **cmp_params)
    assert_close(s_sub, s_sub_ref, **cmp_params)
    assert_close(p, p_ref, **cmp_params)
    assert_close(dp, dp_ref, **cmp_params)
    assert_close(dp_sub, dp_sub_ref, **cmp_params)
    assert_close(ds, ds_ref, **cmp_params)
    assert_close(dq, dq_ref, **cmp_params)
