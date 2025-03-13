# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
import torch

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
        print(dynamic_attention_pipelined(q, k, v, output))

        # CHECK-LABEL:       func.func @dynamic_attention_pipelined
        # CHECK-COUNT-4:        {{.*}} = vector.maskedload {{.*}}
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-2:            {{.*}} = vector.maskedload {{.*}}
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-2:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-2:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-4:            {{.*}} = vector.load {{.*}}
        # CHECK-COUNT-15:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-5:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-4:        vector.maskedstore {{.*}}


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
        print(base_attention_pipelined(q, k, v, output))

        # CHECK-LABEL:       func.func @base_attention_pipelined
        # CHECK:                {{.*}} = scf.for
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-5:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-2:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-1:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-15:            {{.*}} = amdgpu.mfma
        # CHECK-COUNT-5:            {{.*}} = gpu.shuffle xor {{.*}}
        # CHECK-COUNT-1:            {{.*}} = amdgpu.mfma
