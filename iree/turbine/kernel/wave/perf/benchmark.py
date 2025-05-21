# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import torch
import iree.turbine.kernel.wave.nn as wave_nn
from typing import Callable

try:
    from rpdTracerControl import rpdTracerControl
    import sqlite3
    import pandas as pd
except ImportError:
    print("rpdTraceControl not found, skipping profiling")
    exit(1)

try:
    from flash_attn import flash_attn_func

    flash_attn_enabled = True
except:
    flash_attn_enabled = False

DEFAULT_OUTPUT_FILENAME = "trace.rpd"


def benchmark_sdpa(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_k: int,
    head_dim: int,
    num_warmup: int,
    num_iterations: int,
    func: Callable,
    output_filename: str = DEFAULT_OUTPUT_FILENAME,
):
    """Benchmark wave_sdpa with given BHSD shapes.

    Args:
        batch_size: Batch dimension
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        head_dim: Head dimension
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
    """
    device = torch.device("cuda:0")

    # Create input tensors
    query = torch.randn(
        [batch_size, num_heads, seq_len_q, head_dim], device=device, dtype=torch.float16
    )
    key = torch.randn(
        [batch_size, num_heads, seq_len_k, head_dim], device=device, dtype=torch.float16
    )
    value = torch.randn(
        [batch_size, num_heads, seq_len_k, head_dim], device=device, dtype=torch.float16
    )

    # Warmup
    for _ in range(num_warmup):
        _ = func(query, key, value)

    # Synchronize GPU
    torch.cuda.synchronize()

    # Initialize RPD tracer
    rpdTracerControl.setFilename(name=output_filename, append=False)
    tracer = rpdTracerControl()
    tracer.start()

    # Benchmark with profiling
    for _ in range(num_iterations):
        _ = func(query, key, value)
    torch.cuda.synchronize()

    # Stop profiling and get results
    tracer.stop()
    tracer.flush()

    # Calculate statistics
    conn = sqlite3.connect(output_filename)
    df_top = pd.read_sql_query("SELECT * from top", conn)
    df_busy = pd.read_sql_query("SELECT * from busy", conn)
    df_kernel_info = pd.read_sql_query(
        "select s1.string, k.stream, k.gridX, k.gridY, k.gridZ, k.workgroupX, k.workgroupY, k.workgroupZ, s2.string from rocpd_kernelapi k left join rocpd_string s1 on k.api_ptr_id = s1.id left join rocpd_string s2 on k.kernelName_id = s2.id;",
        conn,
    ).drop_duplicates()
    print(df_kernel_info)
    print(df_top)
    conn.close()
    avg_time = df_top["Ave_us"][0] / 1e6

    throughput = (
        4 * batch_size * num_heads * seq_len_q * seq_len_k * head_dim
    ) / avg_time

    print(f"\nBenchmark Results for BHSD shapes:")
    print(f"Batch size: {batch_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Query sequence length: {seq_len_q}")
    print(f"Key sequence length: {seq_len_k}")
    print(f"Head dimension: {head_dim}")
    print(f"Average time per iteration: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput/1e12:.2f} TFLOPs")


def main():
    parser = argparse.ArgumentParser(description="Benchmark wave_sdpa with BHSD shapes")
    parser.add_argument(
        "--output", type=str, default="trace.rpd", help="Path to output trace file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config file"
    )
    parser.add_argument(
        "--num_warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=100, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--fa", action="store_true", help="Benchmark FA instead of Wave, if available"
    )

    args = parser.parse_args()
    global flash_attn_enabled
    if args.fa and not flash_attn_enabled:
        raise ValueError("FA enabled but not installed on system!")

    flash_attn_enabled = args.fa and flash_attn_enabled

    # Load configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Run benchmark for each configuration
    for shape_config in config["shapes"]:
        benchmark_sdpa(
            shape_config["batch_size"],
            shape_config["num_heads"],
            shape_config["seq_len_q"],
            shape_config["seq_len_k"],
            shape_config["head_dim"],
            args.num_warmup,
            args.num_iterations,
            flash_attn_func if flash_attn_enabled else wave_nn.functional.wave_sdpa,
            args.output,
        )


if __name__ == "__main__":
    main()
