# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
import gc
import argparse
from typing import Callable, Sequence
import os
import shlex
import statistics

import torch
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import DeviceType, ProfilerActivity, profile

from iree.turbine.kernel.boo.driver.launch import get_launchable
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [... MIOpenDriver command ...] [--commands-file COMMANDS_FILE]",
        description="""
Run a kernel with the IREE runtime. Command line arguments mirror the
arguments to MIOpenDriver.

Currently supports convolution, layernorm, and matrix multiply.

If COMMANDS_FILE is specified, driver commands are read from the file. Each
line is treated as a separate invocation of the driver, and any additional
command-line arguments are appended to the arguments from the file.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--commands-file", type=str, help="read commands from file")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="file to output timing information in csv format",
    )
    parser.add_argument(
        "--gpu-id",
        default=0,
        type=int,
        help="Indicate a specific gpu device index to run on. Specify '-1' to use all available devices."
        " The index corresponds to a torch.device('cuda:<gpu-id>'). If intending to run on a subset"
        " of available devices, please use the environment variables `CUDA_VISIBLE_DEVICES` or "
        " `ROCR_VISIBLE_DEVICES` along with '--gpu-id=-1'.",
    )
    args, extra_cli_args = parser.parse_known_args()

    if args.commands_file:
        with open(args.commands_file) as f:
            mio_file_args = [
                shlex.split(s) for s in f.readlines() if not s.startswith("#")
            ]
    else:
        mio_file_args = [[]]  # use CLI arguments

    csv_file = open(args.csv if args.csv is not None else os.devnull, "w")
    csv_file.write("arguments,min_time (us)\n")

    runner_parser = argparse.ArgumentParser()
    runner_parser.add_argument("--time", "-t", type=int, help="Enable timing")
    for file_args in mio_file_args:
        driver_args = file_args + extra_cli_args
        timing_args, runner_args = runner_parser.parse_known_args(driver_args)
        func = lambda: run(runner_args, args.gpu_id)
        csv_file.write(shlex.join(driver_args) + ",")
        if not timing_args.time:
            func()
            continue

        try:
            zones, func_name = profile_gpu(func)
        except Exception as exc:
            print(f">>> ERROR: {exc}")
            csv_file.write("N.A.\n")
            continue
        dispatch_zone_names = [n for n in zones.keys() if n.startswith(func_name)]
        if len(dispatch_zone_names) == 0:
            print(">>> FAILED TO COLLECT TIMING INFO")
            csv_file.write("failed to collect timing info\n")
            continue
        for zone_name in dispatch_zone_names:
            times = zones[zone_name]
            s = ">>> "
            s += f"min={min(times):.2f}us "
            s += f"max={max(times):.2f}us "
            s += f"mean={statistics.mean(times):.2f}us "
            s += f"stddev={statistics.stdev(times) if len(times) > 1 else 0:.2f}us"
            if len(dispatch_zone_names) > 1:
                s += f"\t({zone_name})"
            print(s)

        if len(dispatch_zone_names) == 0:
            csv_file.write("no timing info\n")
        elif len(dispatch_zone_names) > 1:
            csv_file.write("multiple dispatches\n")
        else:
            csv_file.write(f"{min(zones[dispatch_zone_names[0]]):.2f}\n")


def run(cli_args: Sequence[str], gpu_id: int):
    print(shlex.join(cli_args))
    key = BooOpRegistry.find_key_from_command(" ".join(cli_args))
    if key is None:
        raise ValueError(
            "unsupported operation kind in "
            + shlex.join(cli_args)
            + ". Supported operations: "
            + str(BooOpRegistry.keys())
        )
    parser_cls = BooOpRegistry.get_parser(key)
    parser = parser_cls.get_miopen_parser()
    parser.add_argument(
        "--iter", type=int, help="Number of iterations to run", default=100
    )
    parser.add_argument(
        "--splat-input-value",
        default=None,
        type=int,
        help="use a splat value for inputs (defaults to random values)",
    )
    args = parser.parse_args(cli_args)
    sig = parser_cls.get_signature(args)
    launchable = get_launchable(sig)

    # get the number of available GPU's
    num_devices = 1 if gpu_id != -1 else torch.cuda.device_count()
    devices = (
        [f"cuda:{gpu_id}"]
        if gpu_id != -1
        else [f"cuda:{i}" for i in range(num_devices)]
    )
    iter_per_device = args.iter // num_devices
    rem_iter = args.iter % num_devices

    # Generate sample args on each GPU.
    per_device_data = [
        sig.get_sample_args(
            seed=10,
            device=device,
            splat_value=args.splat_input_value,
        )
        for device in devices
    ]

    # Determine an iter threshold to pause and collect garbage.
    res_mem_bytes = sig.get_output_size()
    # This is a rough threshold: Mi300x 192 GB memory divided by 2.
    mem_bytes_threshold = 96 * (10**9)
    iter_thresh = int(mem_bytes_threshold // res_mem_bytes)

    results: tuple[torch.Tensor, ...] | torch.Tensor | None = None
    for iter in range(iter_per_device + 1):
        for device_idx, launch_args in enumerate(per_device_data):
            if iter == iter_per_device and device_idx >= rem_iter:
                break
            results = launchable(*launch_args)
        if (iter + 1) % iter_thresh == 0:
            print(f"Synchronizing all devices on iter {iter} and collecting garbage.")
            for i in range(num_devices):
                torch.cuda.synchronize(torch.device(f"cuda:{i}"))
            gc.collect()

    torch.cuda.synchronize()
    if results is None:
        results = ()
    if isinstance(results, torch.Tensor):
        results = (results,)
    for i, result in enumerate(results):
        print(
            f">>> result #{i} shape: {result.shape}; dtype: {result.dtype}; device type: {result.device.type}"
        )

    return sig.func_name


def profile_gpu(func: Callable[[], str]) -> tuple[dict[str, list[int]], str]:
    """Profile 'func' and return the GPU zone execution times, in microseconds."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = func()
    event_list = prof.events()
    assert event_list is not None
    events = defaultdict[str, list[int]](list)
    for event in event_list:
        assert isinstance(event, FunctionEvent)
        if event.device_type == DeviceType.CUDA:
            events[event.name].append(event.self_device_time_total)
    return events, result


if __name__ == "__main__":
    main()
