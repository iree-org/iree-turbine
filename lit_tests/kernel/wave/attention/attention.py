# RUN: python %s | FileCheck %s

import iree.turbine.kernel as tk
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
from iree.turbine.kernel.wave.templates.vanilla_attention import (
    get_bshd_attention_kernel,
    get_vanilla_attention_kernel,
)
from iree.turbine.kernel.wave.templates.attention_common import (
    AttentionShape,
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
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
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

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    base_attention_32x32x8 = wave_compile(options, base_attention_32x32x8)
    print(base_attention_32x32x8.asm)

    # CHECK:            #iree_codegen.translation_info
    # CHECK-SAME:       {llvm_func_attrs = {"amdgpu-waves-per-eu" = "2", "denormal-fp-math-f32" = "preserve-sign"}
    # CHECK-LABEL:      func.func @base_attention_32x32x8
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-8:           {{.*}} = amdgpu.mfma

    # Test for reduction decomposition related to softmax.
    # CHECK-NOT:                arith.maximumf {{.*}}, {{.*}} : vector<16xf32>
    # CHECK-COUNT-30:           arith.maximumf {{.*}}, {{.*}} : f32
    # CHECK:                    {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-2:            arith.maximumf {{.*}}, {{.*}} : vector<1xf32>
    # CHECK:                    arith.addf {{.*}}, {{.*}} : vector<16xf32>
    # CHECK-COUNT-14:           arith.addf {{.*}}, {{.*}} : f32
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
        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
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

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    dynamic_attention_32x32x8 = wave_compile(options, dynamic_attention_32x32x8)
    print(dynamic_attention_32x32x8.asm)

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
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
        func_name="test_vanilla_attention",
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:            {{.*}} = arith.cmpi slt
    # CHECK-COUNT-4:            {{.*}} = arith.select
    # CHECK-COUNT-8:            {{.*}} = arith.addf
    # CHECK-COUNT-8:            {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            {{.*}} = amdgpu.mfma

    # CHECK-LABEL:      func.func @test_vanilla_attention


@run_test
def test_attention_buffer_ops():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        func_name="test_vanilla_attention_buffer_ops",
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention
    # CHECK-COUNT-8:        amdgpu.raw_buffer_load
    # CHECK:                scf.for
    # CHECK-COUNT-4:            amdgpu.raw_buffer_load
    # CHECK:                scf.yield
    # CHECK-COUNT-32:       amdgpu.raw_buffer_store

    # CHECK-LABEL:      func.func @test_vanilla_attention_buffer_ops


@run_test
def test_attention_causal():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False, is_causal=True
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention
    # CHECK:                %[[NEG_INF:.+]] = arith.constant dense<-1.000000e+06> : vector<4xf32>
    # CHECK:                %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-16:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:            {{.*}} = arith.cmpi slt, {{.*}} : vector<4xindex>
    # CHECK-COUNT-8:            {{.*}} = arith.cmpi sge, {{.*}} : vector<4xi64>
    # CHECK-COUNT-8:            {{.*}} = arith.andi {{.*}} : vector<4xi1>
    # CHECK-COUNT-8:            {{.*}} = arith.select %{{.*}}, %[[ZERO]], %[[NEG_INF]] : vector<4xi1>, vector<4xf32>
    # CHECK-COUNT-8:            {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
    # CHECK-COUNT-8:            {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:            {{.*}} = amdgpu.mfma


@run_test
def test_attention_bshd():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _, _ = get_bshd_attention_kernel(
        shape,
        mfma_variant,
        dynamic_dims=False,
        is_causal=False,
        is_custom_mask=True,
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention_custom_mask
    # CHECK-DAG:                %[[INIT_MAX:.+]] = arith.constant dense<-1.000000e+05> : vector<1xf32>
    # CHECK-DAG:                %[[NEG_INF:.+]] = arith.constant dense<-1.000000e+06> : vector<4xf32>
    # CHECK-DAG:                %[[ZEROF:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK:                    {{.*}} = scf.for
    # CHECK-COUNT-16:               {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:                {{.*}} = arith.cmpi sge, {{.*}} : vector<4xindex>
    # CHECK-COUNT-8:                {{.*}} = arith.ori {{.*}} : vector<4xi1>
    # CHECK-COUNT-8:                {{.*}} = arith.select %{{.*}}, %[[ZEROF:.+]], %[[NEG_INF:.+]] : vector<4xi1>, vector<4xf32>
    # CHECK-COUNT-8:                {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
    # CHECK-COUNT-8:                {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-8:                {{.*}} = amdgpu.mfma


@run_test
def test_attention_sliding_window():
    shape = AttentionShape(
        num_query_heads=8,
        num_kv_heads=8,
        query_seq_len=128,
        head_size_kv=128,
        head_size=64,
        kv_seq_len=256,
    )
    mfma_variant = (tkw.MMAType.F32_16x16x16_F16,) * 2
    base_attention, hyperparams, _, _ = get_vanilla_attention_kernel(
        shape, mfma_variant, False, is_causal=True, sliding_window_size=1024
    )

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=True,
    )
    base_attention = wave_compile(options, base_attention)
    print(base_attention.asm)

    # CHECK-LABEL:       func.func @base_attention
    # CHECK:                %[[NEG_INF:.+]] = arith.constant dense<-1.000000e+06> : vector<4xf32>
    # CHECK:                %[[WINDOW_SIZE:.+]] = arith.constant dense<1024> : vector<4xi64>
    # CHECK:                %[[ZERO:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
    # CHECK:                {{.*}} = scf.for
    # CHECK-COUNT-32:           {{.*}} = amdgpu.mfma
    # CHECK-COUNT-4:            {{.*}} = arith.cmpi slt, {{.*}} : vector<4xindex>
    # CHECK-COUNT-8:            {{.*}} = arith.cmpi sge, {{.*}} : vector<4xi64>
    # CHECK-COUNT-8:            {{.*}} = arith.andi {{.*}} : vector<4xi1>
    # This is computing the index difference: m_index - k2_index
    # CHECK-COUNT-8:            {{.*}} = arith.subi {{.*}} : vector<4xi64>
    # And then comparing to the window size: m_index - k2_index <= window_size
    # CHECK-COUNT-8:            {{.*}} = arith.cmpi sle, {{.*}}, %[[WINDOW_SIZE]] : vector<4xi64>
    # CHECK-COUNT-8:            {{.*}} = arith.andi {{.*}} : vector<4xi1>
    # CHECK-COUNT-8:            {{.*}} = arith.select %{{.*}}, %[[ZERO]], %[[NEG_INF]] : vector<4xi1>, vector<4xf32>
    # CHECK-COUNT-8:            {{.*}} = arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
    # CHECK-COUNT-8:            {{.*}} = gpu.shuffle xor {{.*}}
    # CHECK-COUNT-32:           {{.*}} = amdgpu.mfma
