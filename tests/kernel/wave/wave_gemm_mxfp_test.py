import torch
import pytest

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.scheduling.schedule_enums import SchedulingType
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_randn,
    device_randint,
    device_tensor,
    device_zeros,
)
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import (
    ScaledMMAType,
)

from .common.utils import require_e2e, require_cdna4

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(shape):
    M, N, K = shape
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x_high = device_randint(0, 16, (M, K // 2), dtype=torch.uint8)
    x = x_low | x_high << 4
    w_low = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w_high = device_randint(0, 16, (N, K // 2), dtype=torch.uint8)
    w = w_low | w_high << 4
    w = w.T
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8)
    w_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8)
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()

    return x, w, x_scales, w_scales


def generate_gemm_afp8wfp8_inputs(shape):
    M, N, K = shape
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x = device_randn((M, K), dtype=torch.float32).to(torch.float8_e5m2)
    w = device_randn((N, K), dtype=torch.float32).to(torch.float8_e5m2)
    w = w.T
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8)
    w_scales = device_randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8)
    x_scales = x_scales.T
    w_scales = w_scales.T

    return x, w, x_scales, w_scales


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = device_tensor(mxfp4_list, dtype=torch.float32)
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torchScaledGemmMXFP4(x, w, x_scales, w_scales):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T)
    w_f32 = w_f32.T
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


def torchScaledGemmMXFP8(x, w, x_scales, w_scales):
    # First convert the x and w inputs to f32.
    x_f32 = x.to(torch.float32)
    w_f32 = w.to(torch.float32)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.T
    return torch.mm(x_f32, w_f32)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
def testScaledGemmMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)


# BMK @ NK -> BMN represents Linear Layer style BMM.
@require_e2e
@require_cdna4
@pytest.mark.parametrize("batch", [4, 8])
@pytest.mark.parametrize(
    "shape", [(1024, 1024, 1024), (8192, 8192, 8192), (16384, 16384, 16384)]
)
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
def testScaledBatchedGemmMXFP4(
    batch: int,
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
            vector_shapes={B: 0},
        )
    ]

    @tkw.wave(constraints)
    def batched_gemm(
        a: tkl.Memory[B, M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[B, M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_M: 256,
        BLOCK_N: 128,
        BLOCK_K: 256,
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    batched_gemm = wave_compile(options, batched_gemm)

    linearized_shape = (batch * shape[0], shape[1], shape[2])
    flat_x, w, flat_x_scales, w_scales = generate_gemm_afp4wfp4_inputs(linearized_shape)
    w_t = w.T.contiguous()

    x = flat_x.view(batch, shape[0], shape[2] // 2)
    x_scales = flat_x_scales.view(batch, shape[0], shape[2] // 32)
    w_t = w_t.view(shape[1], shape[2] // 2)
    w_scales = w_scales.view(shape[1], shape[2] // 32)
    out = device_zeros(batch, shape[0], shape[1], dtype=torch.float32)

    batched_gemm(x, x_scales, w_t, w_scales, out)
    torch_flat_out = torchScaledGemmMXFP4(flat_x, w, flat_x_scales, w_scales)
    torch_out = torch_flat_out.view(batch, shape[0], shape[1])
    torch.testing.assert_close(torch_out, out)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.NONE])
def testScaledGemmMXFP8(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f8e5m2)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f8e5m2)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 32,
        BLOCK_N: 32,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)

    x, w, x_scales, w_scales = generate_gemm_afp8wfp8_inputs(shape)
    out = device_zeros(x.shape[0], w.shape[1], dtype=torch.float32)

    w_t = w.T.contiguous()
    gemm(x.view(torch.int8), x_scales, w_t.view(torch.int8), w_scales, out)
    torch_out = torchScaledGemmMXFP8(x, w, x_scales, w_scales)
    torch.testing.assert_close(torch_out, out, atol=2e-3, rtol=1e-3, check_dtype=False)


@require_e2e
@require_cdna4
@pytest.mark.parametrize("shape", [(1024, 1024, 1024), (8192, 8192, 8192)])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        ScaledMMAType.F32_16x16x128_F8F6F4,
    ],
)
@pytest.mark.parametrize("enable_scheduling", [SchedulingType.PREFETCH])
def testScaledGemmPrefetechMXFP4(
    shape: tuple[int],
    mfma_variant: ScaledMMAType,
    enable_scheduling: SchedulingType,
):
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [tkw.HardwareConstraint(threads_per_wave=64, mma_type=mfma_variant)]

    @tkw.wave(constraints)
    def prefetch_gemm(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        b_scale: tkl.Memory[N, K / 32, ADDRESS_SPACE, tkl.i8],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a)
            a_reg = tkw.bitcast(a_reg, tkl.f4e2m1fn)
            a_scale_reg = tkw.read(a_scale)
            a_scale_reg = tkw.bitcast(a_scale_reg, tkl.f8e8m0fnu)
            b_reg = tkw.read(b)
            b_reg = tkw.bitcast(b_reg, tkl.f4e2m1fn)
            b_scale_reg = tkw.read(b_scale)
            b_scale_reg = tkw.bitcast(b_scale_reg, tkl.f8e8m0fnu)
            acc = tkw.scaled_mma(a_reg, a_scale_reg, b_reg, b_scale_reg, acc)
            return acc

        tkw.write(repeat, c)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: 256,
        BLOCK_N: 256,
        BLOCK_K: 256,
        M: shape[0],
        N: shape[1],
        K: shape[2],
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
    )
    options = set_default_run_config(options)
    prefetch_gemm = wave_compile(options, prefetch_gemm)

    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=torch.float32)

    w_t = w.T.contiguous()
    prefetch_gemm(x, x_scales, w_t, w_scales, out)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    torch.testing.assert_close(torch_out, out, check_dtype=False)
