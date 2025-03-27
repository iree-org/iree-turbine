# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import ml_dtypes
import numpy
import subprocess
import tempfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import torch
import iree.runtime
from ...support.conversions import TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM
from .compile_options import WaveCompileOptions

BenchmarkResult = namedtuple(
    "BenchmarkResult", "benchmark_name time cpu_time iterations user_counters"
)


def construct_inputs(
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
) -> tuple[list[str], list[tempfile.NamedTemporaryFile]]:
    bench_with_constant_weights = options.bench_with_constant_weights
    tempfiles = []
    inputs = []
    all_inputs = kernel_inputs + kernel_outputs if options.inplace else kernel_inputs
    all_inputs += options.dynamic_symbols_map.values()
    if bench_with_constant_weights:
        for inp in all_inputs:
            if isinstance(inp, torch.Tensor):
                inputs.append(
                    "x".join(
                        [str(x) for x in inp.shape]
                        + [TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM[inp.dtype]]
                    )
                )
            elif isinstance(inp, int):
                inputs.append(f"1xi32={inp}")
            else:
                raise NotImplementedError("Unsupported input type.")
    else:
        for inp in all_inputs:
            if isinstance(inp, torch.Tensor):
                inp = inp.cpu()
                if inp.dtype == torch.bfloat16:
                    inp = (
                        inp.view(dtype=torch.uint16)
                        .numpy()
                        .view(dtype=ml_dtypes.bfloat16)
                    )
                else:
                    inp = inp.numpy()
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tf:
                    numpy.save(tf, inp)
                    tempfiles.append(tf)
                    inputs.append("@" + tf.name)
            elif isinstance(inp, int):
                inputs.append(f"1xi32={inp}")
            else:
                raise NotImplementedError("Unsupported input type.")

    input_prefix = "--input="
    inputs = [input_prefix + inp for inp in inputs]
    return inputs, tempfiles


def parse_benchmark_results(out: bytes) -> list[BenchmarkResult]:
    # Grab individual results by line (skip header lines)
    bench_lines = out.decode().split("\n")[3:]
    benchmark_results = []
    for line in bench_lines:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark_name = split[0]
        time = " ".join(split[1:3])
        cpu_time = " ".join(split[3:5])
        iterations = split[5]
        user_counters = None
        if len(split) > 5:
            user_counters = split[6]
        benchmark_results.append(
            BenchmarkResult(
                benchmark_name=benchmark_name,
                time=time,
                cpu_time=cpu_time,
                iterations=iterations,
                user_counters=user_counters,
            )
        )
    return benchmark_results


def create_trace_json():
    data = {
        "jobs": [
            {
                "advanced_thread_trace": "true",
                "att_parse": "trace",
                "att_target_cu": 1,
                "att_shader_engine_mask": "0x1",
                "att_simd_select": "0xF",
                "att_buffer_size": "0x6000000",
            }
        ]
    }
    with open("input.json", "w") as f:
        json.dump(data, f)


def populate_trace_args(prefix: list[str], capture_trace_dir: str) -> list[str]:
    create_trace_json()
    prefix += ["rocprofv3", "-i", "input.json", "-d"]
    prefix += [capture_trace_dir]
    prefix += ["--"]
    return prefix


def benchmark_module(
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    flatbuffer: bytes,
    entry_function: str,
    timeout=None,
    **kwargs,
):

    prefix = []
    if options.capture_trace:
        prefix = populate_trace_args(prefix, options.capture_trace)
    args = prefix + [iree.runtime.benchmark_exe()]
    args.append(f"--function={entry_function}")
    for k in kwargs:
        v = kwargs[k]
        args.append(f"--{k}={v}")
    inputs, tempfiles = construct_inputs(options, kernel_inputs, kernel_outputs)
    args += inputs
    args.append("--module=-")
    args.append(f"--device={options.device}")

    try:
        benchmark_process = subprocess.run(
            args=args,
            input=flatbuffer,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Benchmark timed out after {timeout} seconds")
    out = benchmark_process.stdout
    err = benchmark_process.stderr

    err = err.decode()
    if "INVALID_ARGUMENT;" in err:
        raise ValueError("Invalid inputs specified for benchmarking")

    # In the event benchmarking runs but encounteres an internal error,
    # return the internal error instead of benchmark results.
    if "INTERNAL; CUDA driver error" in str(out):
        raise ValueError(str(out))

    benchmark_results = parse_benchmark_results(out)
    for file in tempfiles:
        Path.unlink(file.name)

    return benchmark_results
