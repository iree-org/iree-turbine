# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.templates.decode_attention import (
    get_decode_attention_kernels,
)
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    get_paged_decode_attention_kernels,
    paged_decode_attention_shape,
)
import torch
import sympy
import math

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


@run_test
def test_evoformer():
    # B, BN, K2, H, K1, M, N
    shape = (1, 256, 256, 4, 32, 256, 32)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 3)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 4)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, BN: 0, H: 0, M: 16, N: 16},
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
    # [B, BN, K2, H, N] -> [B, BN, H, N, K2]
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
    def evoformer(
        q: tkl.Memory[B, BN, M, H, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, BN, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, BN, K2, H, N, ADDRESS_SPACE, tkl.f16],
        mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        c_reg = tkl.Register[B, BN, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, BN, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, BN, H, M, tkl.f32](-1e6)

        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
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
            k_reg = tkw.read(
                k, mapping=k_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
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
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        casted = tkw.cast(res, tkl.f16)
        tkw.write(
            casted, c, mapping=o_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        B: shape[0],
        BN: shape[1],
        K2: shape[2],
        H: shape[3],
        K1: shape[4],
        M: shape[5],
        N: shape[6],
        BLOCK_B: 1,
        BLOCK_BN: 1,
        BLOCK_H: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(evoformer(q, k, v, output).module_op)

        # CHECK:            func.func @evoformer
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-5:            {{.*}} = vector.load
        # CHECK-COUNT-1:            vector.store {{.*}}
        # CHECK-COUNT-4:            {{.*}} = vector.load
        # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma
        # CHECK-COUNT-2:            {{.*}} = vector.load
        # CHECK-COUNT-2:            {{.*}} = arith.extf
        # CHECK-COUNT-4:            {{.*}} = arith.addf
        # CHECK-COUNT-4:            {{.*}} = vector.load
        # CHECK-COUNT-4:            {{.*}} = arith.extf
        # CHECK-COUNT-4:            {{.*}} = arith.addf


# This test sets all the dimensions except K1 to be dynamic.
# The reason why we can't set K1 to be dynamic is because K1 is the
# tile size we use for expanding the K1 MMA. We could set K1 to be
# dynamic if we tiled the K1 dimension with a tile size of BLOCK_K1.
@run_test
def test_dynamic_attention_pipelined():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        )
    ]

    constraints += [tkw.Assumption(K2 > 4 * BLOCK_K2)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def dynamic_attention_pipelined(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        K1: shape[3],
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=True,
        use_scheduling_barriers=False,
        dynamic_symbols=(B, M, N, K2),
        dynamic_symbol_map={
            B: shape[0],
            M: shape[1],
            N: shape[2],
            K2: shape[4],
        },
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(dynamic_attention_pipelined(q, k, v, output).module_op)

        # CHECK-LABEL:       func.func @dynamic_attention_pipelined
        # CHECK-COUNT-2:        {{.*}} = vector.maskedload {{.*}}
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-2:            {{.*}} = vector.maskedload {{.*}}
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-2:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-2:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-6:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-11:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-5:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-16:       vector.maskedstore {{.*}}


@run_test
def test_attention_pipelined():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention_pipelined(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=True,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(base_attention_pipelined(q, k, v, output).module_op)

        # CHECK-LABEL:       func.func @base_attention_pipelined
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-19:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-5:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma


@run_test
def test_flash_decoding():
    shape = (8, 128, 128, 64, 256)
    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    use_dynamic_dims = True
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_map_0,
        dynamic_symbols_1,
        dynamic_symbols_map_1,
    ) = get_decode_attention_kernels(shape, mfma_variant, use_dynamic_dims)

    torch.manual_seed(0)
    q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
    k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
    v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
    logits = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
    logits_max = torch.zeros(shape[0], shape[1], dtype=torch.float32)
    output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols_0,
        dynamic_symbols_map=dynamic_symbols_map_0,
    ):
        print(phase_0(q, k, v, logits, logits_max).module_op)

    # CHECK:            func.func @phase_0
    # CHECK-NOT:               {{.*}} = scf.for
    # CHECK-COUNT-9:           {{.*}} = vector.maskedload
    # CHECK-COUNT-1:           vector.store
    # CHECK-COUNT-4:           {{.*}} = vector.load
    # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:           {{.*}} = gpu.shuffle
    # CHECK-COUNT-2:           {{.*}} = arith.subf
    # CHECK-COUNT-2:           {{.*}} = math.exp2
    # CHECK-COUNT-2:           {{.*}} = arith.subf
    # CHECK-COUNT-2:           {{.*}} = math.exp2
    # CHECK-COUNT-4:           {{.*}} = gpu.shuffle
    # CHECK-COUNT-2:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-2:           {{.*}} = arith.divf
    # CHECK-COUNT-2:           {{.*}} = math.log2
    # CHECK-COUNT-18:          vector.maskedstore

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
        dynamic_symbols=dynamic_symbols_1,
        dynamic_symbols_map=dynamic_symbols_map_1,
    ):
        print(phase_1(logits, logits_max, output).module_op)

    # CHECK:            func.func @phase_1
    # CHECK:               {{.*}} = scf.for
    # CHECK-COUNT-2:           {{.*}} = vector.maskedload
    # CHECK-COUNT-1:           {{.*}} = arith.maximumf
    # CHECK-COUNT-1:           {{.*}} = arith.subf
    # CHECK-COUNT-1:           {{.*}} = math.exp2
    # CHECK-COUNT-1:           {{.*}} = arith.subf
    # CHECK-COUNT-1:           {{.*}} = math.exp2
    # CHECK-COUNT-1:           {{.*}} = arith.mulf
    # CHECK-COUNT-1:           {{.*}} = arith.addf
    # CHECK-COUNT-2:           {{.*}} = arith.mulf
    # CHECK-COUNT-1:           {{.*}} = arith.addf
    # CHECK-COUNT-1:      {{.*}} = arith.divf
    # TODO: Remove vector.scatter when optimizing for performance
    # CHECK-COUNT-1:      vector.scatter


@run_test
def test_attention_32x32x8():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_32x32x8_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 32, N: 32},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention_32x32x8(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }

    compile_config = {"waves_per_eu": 2, "denorm_fp_math_f32": "preserve-sign"}
    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        compile_config=compile_config,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(base_attention_32x32x8(q, k, v, output).module_op)

        # CHECK-DAG:        #iree_codegen.translation_info
        # CHECK-SAME:       {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}
        # CHECK-LABEL:      func.func @base_attention_32x32x8
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma

        # Test for reduction decomposition related to softmax.
        # CHECK-NOT:                arith.maximumf {{.*}}, {{.*}} : vector<16xf32>
        # CHECK-COUNT-30:           arith.maximumf {{.*}}, {{.*}} : vector<1xf32>
        # CHECK:                    {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-2:            arith.maximumf {{.*}}, {{.*}} : vector<1xf32>
        # CHECK:                    arith.addf {{.*}}, {{.*}} : vector<16xf32>
        # CHECK-COUNT-14:           arith.addf {{.*}}, {{.*}} : vector<1xf32>
        # CHECK:                    {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-2:            arith.addf {{.*}}, {{.*}} : vector<1xf32>

        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [4], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [8], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [12], sizes = [4], strides = [1]}
        # CHECK-COUNT-4:            {{.*}} = amdgpu.mfma
        # CHECK:                    scf.yield
        # CHECK-COUNT-4:            vector.store {{.*}}: memref<8x128x128xf32{{.*}}>, vector<4xf32>


@run_test
def test_dynamic_attention_32x32x8():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]
    constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    mfma_variant = tkw.MMAType.F32_32x32x8_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 32, N: 32},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def dynamic_attention_32x32x8(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 64,
        K1: shape[3],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }

    # Set up dynamic parameters
    dynamic_symbols_map = {}
    dynamic_symbols_map[B] = shape[0]
    dynamic_symbols_map[M] = shape[1]
    dynamic_symbols_map[N] = shape[2]
    dynamic_symbols_map[K2] = shape[4]
    dynamic_symbols = [M, N, B, K2]

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(dynamic_attention_32x32x8(q, k, v, output).module_op)

        # CHECK-LABEL:      func.func @dynamic_attention_32x32x8
        # CHECK-DAG:            %[[IOTA:.+]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
        # CHECK-COUNT-2:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [0], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [4], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [8], sizes = [4], strides = [1]}
        # CHECK:                    {{.*}} = vector.extract_strided_slice {{.*}} {offsets = [12], sizes = [4], strides = [1]}
        # CHECK-COUNT-8:            {{.*}} = amdgpu.mfma
        # CHECK:                    scf.yield

        # Check for mask generation and masked stor:
        # CHECK:                %[[INDICES:.+]] = arith.addi %{{.*}}, %[[IOTA]] overflow<nsw, nuw> : vector<4xindex>
        # CHECK:                %[[BOUNDS:.+]] = vector.splat %{{.*}} : vector<4xindex>
        # CHECK:                %[[SLT:.+]] = arith.cmpi slt, %[[INDICES]], %[[BOUNDS]] : vector<4xindex>
        # CHECK:                %[[MASK:.+]] = arith.andi %{{.*}}, %[[SLT]] : vector<4xi1>
        # CHECK:                vector.maskedstore %{{.*}}[{{.*}}], %[[MASK]], %{{.*}} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, vector<4xi1>, vector<4xf32>
        # CHECK-COUNT-3:        vector.maskedstore {{.*}} : memref<?x?x?xf32, strided<[?, ?, 1], offset: ?>>, vector<4xi1>, vector<4xf32>


@run_test
def test_attention():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(base_attention(q, k, v, output).module_op)

        # CHECK-LABEL:       func.func @base_attention
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
        # CHECK-COUNT-8:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            {{.*}} = amdgpu.mfma


@run_test
def test_attention_bias():
    shape = (8, 128, 128, 64, 256)
    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            mma_type=mfma_variant,
            vector_shapes={B: 0, M: 16, N: 16},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    mapping = tkw.IndexMapping(
        num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
    )

    @tkw.wave(constraints)
    def base_attention_bias(
        q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[B, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f32],
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
        ) -> (
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, M, tkl.f32],
            tkl.Register[B, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[B, N, K, tkl.f16]
            k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[B, N, M, tkl.f32]
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            x_j = x_j + bias_reg
            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        tkw.write(res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        B: shape[0],
        M: shape[1],
        N: shape[2],
        K1: shape[3],
        K2: shape[4],
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
    }

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        torch.manual_seed(0)
        q = torch.randn(shape[0], shape[1], shape[3], dtype=torch.float16)
        k = torch.randn(shape[0], shape[4], shape[3], dtype=torch.float16)
        v = torch.randn(shape[0], shape[4], shape[2], dtype=torch.float16)
        output = torch.zeros(shape[0], shape[1], shape[2], dtype=torch.float32)
        print(base_attention_bias(q, k, v, output).module_op)

        # CHECK:            func.func @base_attention_bias
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load
        # CHECK-COUNT-4:            {{.*}} = arith.addf
        # CHECK-COUNT-8:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-8:            {{.*}} = amdgpu.mfma


@run_test
def test_paged_flash_decoding():
    shape = paged_decode_attention_shape(
        num_query_heads=128,
        num_kv_heads=4,
        head_size=32,
        head_size_kv=32,
        block_size=64,
        num_seqs=8,
        kv_lens=100,
    )
    num_kv_splits = 8
    k_shape = (shape.num_seqs, shape.kv_lens, shape.num_kv_heads, shape.head_size)
    v_shape = (shape.num_seqs, shape.kv_lens, shape.num_kv_heads, shape.head_size_kv)
    q_shape = (shape.num_seqs, shape.num_query_heads, shape.head_size)
    o_shape = (shape.num_seqs, shape.num_query_heads, shape.head_size_kv)
    logits_shape = (num_kv_splits, shape.num_seqs, shape.head_size_kv, shape.kv_lens)
    logits_max_shape = (num_kv_splits, shape.num_seqs, shape.head_size_kv)
    block_table_shape = (shape.num_seqs, shape.kv_lens)
    mfma_variant = tkw.MMAType.F32_16x16x16_F16
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
    ) = get_paged_decode_attention_kernels(
        shape, mfma_variant, num_kv_splits, k_shape, v_shape, block_table_shape
    )

    torch.manual_seed(0)
    q = torch.randn(q_shape, dtype=torch.float16)
    k = torch.randn(k_shape, dtype=torch.float16)
    v = torch.randn(v_shape, dtype=torch.float16)
    logits = torch.zeros(logits_shape, dtype=torch.float32)
    logits_max = torch.zeros(logits_max_shape, dtype=torch.float32)
    output = torch.zeros(o_shape, dtype=torch.float16)

    with tk.gen.TestLaunchContext(
        hyperparams_0,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        print(phase_0(q, k, v, logits, logits_max).module_op)

    # CHECK:                func.func @phase_0
    # CHECK-COUNT-4:           vector.load
    # CHECK:                   scf.for
    # CHECK-COUNT-3:                vector.maskedload
    # CHECK-COUNT-8:                vector.store
    # CHECK-COUNT-8:                vector.load
    # CHECK-COUNT-8:               amdgpu.mfma
    # CHECK-COUNT-2:                gpu.shuffle
    # CHECK-COUNT-1:                arith.subf
    # CHECK-COUNT-1:                math.exp2
    # CHECK-COUNT-4:                arith.subf
    # CHECK-COUNT-4:                math.exp2
    # CHECK-COUNT-2:                gpu.shuffle
    # TODO: Remove gathers for performance
    # CHECK-COUNT-2:                vector.gather
    # CHECK-COUNT-8:                vector.store
    # CHECK-COUNT-8:                vector.load
    # CHECK-COUNT-8:                amdgpu.mfma
    # CHECK-COUNT-1:          arith.divf
    # CHECK-COUNT-1:          math.log2
    # CHECK-COUNT-9:         vector.store

    with tk.gen.TestLaunchContext(
        hyperparams_1,
        canonicalize=True,
        run=False,
        run_bench=False,
        schedule=False,
        use_scheduling_barriers=False,
    ):
        print(phase_1(logits, logits_max, output).module_op)

    # CHECK:            func.func @phase_1
    # CHECK:               scf.for
    # CHECK-COUNT-2:           vector.load
    # CHECK-COUNT-1:           arith.maximumf
    # CHECK-COUNT-1:           arith.subf
    # CHECK-COUNT-1:           math.exp2
    # CHECK-COUNT-1:           arith.subf
    # CHECK-COUNT-1:           math.exp2
    # CHECK-COUNT-1:           arith.mulf
    # CHECK-COUNT-1:           arith.addf
    # CHECK-COUNT-2:           arith.mulf
    # CHECK-COUNT-1:           arith.addf
    # CHECK-COUNT-1:      arith.divf
    # CHECK-COUNT-1:      vector.store
