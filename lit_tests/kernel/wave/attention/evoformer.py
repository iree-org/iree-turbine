# RUN: python %s | FileCheck %s

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile

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

        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, BN, H, M, tkl.f32],
            partial_sum: tkl.Register[B, BN, H, M, tkl.f32],
            acc: tkl.Register[B, BN, H, N, M, tkl.f32],
        ) -> tuple[
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, N, M, tkl.f32],
        ]:
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

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    evoformer = wave_compile(options, evoformer)
    print(evoformer.asm)

    # CHECK:            func.func @evoformer
    # CHECK:                {{.*}} = scf.for
    # CHECK:                    {{.*}} = vector.load
    # CHECK:                    vector.store {{.*}}
    # CHECK:                    {{.*}} = vector.maskedload
    # CHECK:                    vector.store {{.*}}
    # CHECK:                    amdgpu.lds_barrier
    # CHECK-COUNT-16:           {{.*}} = memref.load
    # CHECK-COUNT-4:            {{.*}} = vector.load
    # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-2:            {{.*}} = vector.load
    # CHECK-COUNT-2:            {{.*}} = arith.extf
    # CHECK-COUNT-4:            {{.*}} = arith.addf
    # CHECK-COUNT-4:            {{.*}} = vector.load
    # CHECK-COUNT-4:            {{.*}} = arith.extf
    # CHECK-COUNT-4:            {{.*}} = arith.addf
