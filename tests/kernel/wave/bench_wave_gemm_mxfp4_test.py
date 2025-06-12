import argparse
import sys
import torch
import triton
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from op_tests.triton_tests.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs


import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.constraints import (
    ScaledMMAType,
)

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def get_mxfp4_gemm(shape):
    mfma_variant = ScaledMMAType.F32_16x16x128_F8F6F4
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    K_SCALE = tkl.sym.K_SCALE
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

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(4, 2, 1), mma_type=mfma_variant
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    b_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, K: j}, outputs={N: i, K: j}
    )
    # b: tkl.Memory[K/2, N, ADDRESS_SPACE, tkl.i8],
    # b: tkl.Memory[N, K/2, ADDRESS_SPACE, tkl.i8],

    @tkw.wave(constraints)
    def gemm_afp4_wfp4_wave(
        a: tkl.Memory[M, K / 2, ADDRESS_SPACE, tkl.i8],
        a_scale: tkl.Memory[M, K / 32, ADDRESS_SPACE, tkl.i8],
        b: tkl.Memory[N, K / 2, ADDRESS_SPACE, tkl.i8],
        # b: tkl.Memory[K / 2, N, ADDRESS_SPACE, tkl.i8],
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
            b_reg = tkw.read(b, mapping=b_mapping)
            # b_reg = tkw.read(b)
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
        K_SCALE: shape[2] // 32,
    }
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=SchedulingType.PREFETCH,
        use_buffer_load_ops=True,
        dump_intermediates="./inter",
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm_afp4_wfp4_wave)
    return gemm


def get_x_vals():
    x_vals = [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
    ]
    return x_vals


def run_benchmark(args):
    x_names = ["M", "N", "K"]
    if args.shape:
        x_vals_list = [args.shape]
    else:
        x_vals_list = get_x_vals()

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = ["Triton"]
    line_vals = ["triton"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name="GEMM MXFP4 x MXFP4 Benchmark",
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_gemm_afp4wfp4_blockscale(M, N, K, metric, provider):
        c_dtype = torch.bfloat16
        x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs(M, N, K)
        # flops
        flops = 2.0 * M * N * K
        # memory transfer
        mem_read = x.numel() * x.element_size() + w.numel() * w.element_size()
        mem_read += (
            x_scale.numel() * x_scale.element_size()
            + w_scale.numel() * w_scale.element_size()
        )
        mem_write = (M * N) * 2  # TODO: Fix for c_dtype != bf16
        mem = mem_read + mem_write
        out = torch.empty(x.shape[0], w.shape[1], device=x.device, dtype=c_dtype)

        # ms = triton.testing.do_bench(
        #     lambda: gemm_afp4wfp4(x, w, out, x_scale, w_scale, c_dtype),
        #     warmup=25,
        #     rep=100,
        # )
        shape = (M, N, K)
        gemm = get_mxfp4_gemm(shape)
        w_t = w.T.contiguous()
        wave_out = torch.empty(
            x.shape[0], w.shape[1], device=x.device, dtype=torch.float32
        )
        ms = triton.testing.do_bench(
            lambda: gemm(x, x_scale, w_t, w_scale, wave_out),
            warmup=25,
            rep=100,
        )

        # Return exactly one scalar depending on which metric is active
        if metric == "time":
            return ms
        elif metric == "throughput":
            tflops = flops / ms * 1e-9
            return tflops
        elif metric == "bandwidth":
            bandwidth = mem / (ms * 1e-3) * 1e-9  # GB/s
            return bandwidth
        else:
            raise ValueError("Unknown metric: " + metric)

    bench_gemm_afp4wfp4_blockscale.run(save_path=".", print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 x MXFP4 GEMM",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-M",
        type=int,
        default=4096,
        help="M dim of model benchmark if only one model is under test",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["time", "throughput", "bandwidth"],
        default="throughput",
        help="metric to plot",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
