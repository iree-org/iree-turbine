# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from contextlib import nullcontext
import gc
import argparse
from typing import Callable, Sequence, Literal
import os
import shlex
import statistics

import torch
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import DeviceType, ProfilerActivity, profile

from iree.turbine.kernel.boo.exports.signature import OpSignature
from iree.turbine.kernel.boo.driver.launch import get_launchable
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

ZoneData = dict[str, list[float]]
ZoneStats = dict[Literal["min", "max", "mean", "stddev", "count"], float | int]
ZoneStatsSummary = dict[str, ZoneStats]


def _get_main_driver_parser() -> argparse.ArgumentParser:
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
        "--reference-backend",
        type=str,
        choices=["torch", "torch-compile"],
        default=None,
        required=False,
    )
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
    parser.add_argument(
        "--splat-input-value",
        default=None,
        type=int,
        help="Use a splat value for inputs (defaults to random values).",
    )
    return parser


def _get_timing_parser() -> argparse.ArgumentParser:
    """This parser separates out timing-specific args from miopen commands."""
    timing_parser = argparse.ArgumentParser()
    timing_parser.add_argument("--time", "-t", type=int, help="Enable timing")
    timing_parser.add_argument(
        "--iter", type=int, help="Number of iterations to run", default=100
    )
    return timing_parser


def main():
    # Parse input cli args into global driver args and miopen-style commands.
    driver_parser = _get_main_driver_parser()
    meta_args, extra_cli_args = driver_parser.parse_known_args()
    if meta_args.commands_file:
        with open(meta_args.commands_file) as f:
            mio_args = [
                shlex.split(s) + extra_cli_args
                for s in f.readlines()
                if not s.startswith("#")
            ]
    else:
        mio_args = [extra_cli_args]  # use CLI arguments

    # Check the reference backend
    ref_backend = meta_args.reference_backend
    # TODO: Add ability to benchmark against torch-compile (inductor).
    if ref_backend == "torch-compile":
        raise NotImplementedError(
            "Comparing against torch-compiled reference not yet implemented."
        )

    # Setup a csv output file with headers.
    csv_file = open(meta_args.csv if meta_args.csv is not None else os.devnull, "w")
    csv_headers = ["arguments", "iree min time (us)"]
    if ref_backend is not None:
        csv_headers.append(f"{ref_backend} min time (us)")
    csv_file.write(",".join(csv_headers))

    timing_parser = _get_timing_parser()

    devices = _get_devices(meta_args.gpu_id)

    for driver_args in mio_args:
        command = shlex.join(driver_args)
        print(f"\n>>> {command}\n")
        timing_args, runner_args = timing_parser.parse_known_args(driver_args)
        csv_file.write(shlex.join(driver_args) + ",")
        signature = BooOpRegistry.parse_command(shlex.join(runner_args))

        if signature is None:
            print(f">>> Boo op registry failed to parse {shlex.join(runner_args)}.")
            csv_file.write("N.A.\n")
            continue

        sample_inputs = _get_sample_args(
            signature, meta_args.splat_input_value, devices
        )
        launchable = get_launchable(signature)
        for device in devices:
            launchable.preload(device)
        output_num_bytes = signature.get_output_size()

        profile_context = (
            profile(activities=[ProfilerActivity.CUDA])
            if timing_args.time
            else nullcontext()
        )

        ### IREE run ###
        try:
            with profile_context as iree_prof:
                run(launchable, timing_args.iter, output_num_bytes, sample_inputs)
        except Exception as exc:
            print(f">>> ERROR: {exc}")
            csv_file.write("N.A.\n")
            continue

        if not timing_args.time:
            csv_file.write("untimed\n")
            continue

        iree_zones = _extract_zones(iree_prof)

        if len(iree_zones.keys()) == 0:
            print(">>> FAILED TO COLLECT TIMING INFO")
            csv_file.write("failed to collect timing info\n")
            continue

        # Get iree stats and print.
        iree_results = _get_zone_stats(iree_zones)
        _print_zone_stats(iree_results)

        iree_theoretical_min = _get_theoretical_min_time(iree_results)
        print(f">>> Per-launch GPU min time (IREE): {iree_theoretical_min:.2f}us")
        csv_file.write(f"{iree_theoretical_min:.2f}")

        if not ref_backend:
            csv_file.write("\n")
            continue

        ref_func = signature.get_nn_module(use_custom=False)

        ### Reference run ###
        print(f">>> Running with reference backend {ref_backend}.")
        try:
            with profile_context as ref_prof:
                run(ref_func, timing_args.iter, output_num_bytes, sample_inputs)
        except Exception as exc:
            print(f">>> ERROR during reference run: {exc}")
            csv_file.write(",N.A.\n")
            continue

        ref_zones = _extract_zones(ref_prof)
        if ref_zones is None or len(ref_zones) == 0:
            print(f">>> Could not profile reference run.")
            csv_file.write(",N.A.\n")
            continue

        ref_results = _get_zone_stats(ref_zones)
        _print_zone_stats(ref_results)
        ref_theoretical_min = _get_theoretical_min_time(ref_results)
        csv_file.write(f",{ref_theoretical_min:.2f}\n")
        print(
            f">>> Per-launch GPU min time ({ref_backend}): {ref_theoretical_min:.2f}us"
        )


def run(
    func: Callable,
    iter: int,
    output_num_bytes: int,
    per_device_args: Sequence[tuple[torch.Tensor, ...]],
) -> None:
    """Distributes `iter`-many applications of `func` to `per_device_args`."""
    num_devices = len(per_device_args)
    iter_per_device = iter // num_devices
    rem_iter = iter % num_devices
    # This is a rough threshold: Mi300x 192 GB memory divided by 2.
    mem_bytes_threshold = 96 * (10**9)
    iter_thresh = int(mem_bytes_threshold // output_num_bytes)

    results: tuple[torch.Tensor, ...] | torch.Tensor | None = None
    for iter in range(iter_per_device + 1):
        for device_idx, launch_args in enumerate(per_device_args):
            if iter == iter_per_device and device_idx >= rem_iter:
                break
            results = func(*launch_args)
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
    return


def _get_devices(gpu_id: int) -> list[torch.device]:
    """Returns a list of torch.device to test on."""
    num_devices = 1 if gpu_id != -1 else torch.cuda.device_count()
    devices = (
        [torch.device(f"cuda:{gpu_id}")]
        if gpu_id != -1
        else [torch.device(f"cuda:{i}") for i in range(num_devices)]
    )
    return list(devices)


def _get_sample_args(
    sig: OpSignature, splat_input_value: None | int, devices: list[torch.device]
) -> list[tuple[torch.Tensor, ...]]:
    """Generates sample args on each device."""
    per_device_data = [
        sig.get_sample_args(
            seed=10,
            device=device,
            splat_value=splat_input_value,
        )
        for device in devices
    ]
    return per_device_data


def _get_theoretical_min_time(results: ZoneStatsSummary) -> float:
    """Computes a theoretical min time by summing the min time from each zone with multiplicity."""
    total = 0.0
    count: None | int = None
    for stats in results.values():
        if count is None:
            count = int(stats["count"])
        else:
            count = min(count, int(stats["count"]))

        total += stats["min"] * stats["count"]
    assert count is not None and count > 0
    return total / count


def _extract_zones(prof: profile | None) -> ZoneData:
    """Profile 'func' and return the GPU zone execution times, in microseconds."""
    events: ZoneData = defaultdict[str, list[float]](list)
    if prof is None:
        return events
    event_list = prof.events()
    assert event_list is not None
    for event in event_list:
        assert isinstance(event, FunctionEvent)
        if event.device_type == DeviceType.CUDA:
            events[event.name].append(event.self_device_time_total)
    return events


def _get_zone_stats(zones: ZoneData) -> ZoneStatsSummary:
    """Get statistics for each zone as a dictionary."""
    results: ZoneStatsSummary = {}
    for zone_name, times in zones.items():
        min_time = min(times)
        max_time = min(times)
        mean_time = statistics.mean(times)
        stddev = statistics.stdev(times) if len(times) > 1 else 0
        count = len(times)
        results[zone_name] = {
            "min": min_time,
            "max": max_time,
            "mean": mean_time,
            "stddev": stddev,
            "count": count,
        }
    return results


def _print_zone_stats(results: ZoneStatsSummary) -> None:
    """Prints a ZoneStatsSummary."""
    for zone_name, stats in results.items():
        s = ">>> "
        s += f"{zone_name}\n>>>\t"
        decorate_val = lambda val: (
            f"{val:.2f}us" if isinstance(val, float) else str(val)
        )
        s += " ".join([f"{key}={decorate_val(value)}" for key, value in stats.items()])
        print(s)


if __name__ == "__main__":
    main()
