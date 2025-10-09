# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import defaultdict
from contextlib import nullcontext
import gc
import argparse
from typing import Callable, Sequence, NamedTuple
import os
import shlex
import statistics
from functools import partial

import torch
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import DeviceType, ProfilerActivity, profile

from iree.turbine.kernel.boo.exports.signature import OpSignature
from iree.turbine.kernel.boo.driver.launch import get_launchable
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry
from iree.turbine.kernel.boo.driver.utils import get_timing_parser
from iree.turbine.runtime.device import get_device_from_torch

ZoneData = dict[str, list[float]]

ALL_STATS = ["min", "max", "mean", "stddev", "num_dispatches"]


class ZoneStats(NamedTuple):
    min: float | str = "N.A."
    max: float | str = "N.A."
    mean: float | str = "N.A."
    stddev: float | str = "N.A."
    num_dispatches: int | str = "N.A."


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
        choices=[k for k in BACKEND_TO_FUNC_GENERATOR.keys() if k != DEFAULT_BACKEND],
        action="append",
        default=[],
        required=False,
        help="Choose reference backends to compare performance against.",
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
        type=float,
        help="Use a splat value for inputs (defaults to random values).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Print command/output on STDOUT.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_false",
        dest="verbose",
        help="Disable printing command/output on STDOUT.",
    )
    return parser


def main():
    # Parse input cli args into global driver args and miopen-style commands.
    driver_parser = _get_main_driver_parser()
    meta_args, extra_cli_args = driver_parser.parse_known_args()
    # Default to verbose terminal output if we're not writing to a file.
    if meta_args.verbose is None:
        meta_args.verbose = meta_args.csv is None
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
    ref_backends = meta_args.reference_backend

    # Setup a csv output file with headers.
    csv_stats = ALL_STATS
    backends = [DEFAULT_BACKEND] + ref_backends
    csv_file = open(meta_args.csv if meta_args.csv is not None else os.devnull, "w")
    csv_headers = ["arguments"]
    for b in backends:
        for stat in csv_stats:
            csv_headers.extend([f"{b} {stat}"])
    csv_file.write(",".join(csv_headers) + "\n")

    timing_parser = get_timing_parser()

    devices = _get_devices(meta_args.gpu_id)
    testCount = 0

    for driver_args in mio_args:
        testCount = testCount + 1
        command = shlex.join(driver_args)
        if meta_args.verbose:
            print(f"\n>>> {command}\n")
        else:
            print("Running test :", testCount)
        timing_args, runner_args = timing_parser.parse_known_args(driver_args)
        csv_file.write(shlex.join(driver_args) + ",")
        signature = BooOpRegistry.parse_command(shlex.join(runner_args))

        if signature is None:
            if meta_args.verbose:
                print(f">>> Boo op registry failed to parse {shlex.join(runner_args)}.")
            csv_file.write("N.A.\n")
            continue

        sample_inputs = _get_sample_args(
            signature, meta_args.splat_input_value, devices
        )
        output_num_bytes = signature.get_output_size()

        for backend in backends:
            _func = BACKEND_TO_FUNC_GENERATOR[backend](signature)
            try:
                prof = run(
                    _func,
                    timing_args,
                    output_num_bytes,
                    sample_inputs,
                    devices,
                    meta_args.verbose,
                )
            except Exception as exc:
                if meta_args.verbose:
                    print(f">>> ERROR: {exc}")
                csv_file.write("N.A.," * len(csv_stats))
                continue

            if not timing_args.time:
                csv_file.write("untimed," * len(csv_stats))
                continue

            zones = _extract_zones(prof)

            if len(zones.keys()) == 0:
                if meta_args.verbose:
                    print(">>> FAILED TO COLLECT TIMING INFO")
                csv_file.write("failed to collect timing info," * len(csv_stats))
                continue

            # Get iree stats and print.
            results = _get_zone_stats(zones)
            if meta_args.verbose:
                _print_zone_stats(results)

            aggregate_stats = get_aggregate_stats(csv_stats, results, timing_args.iter)
            if meta_args.verbose:
                print(
                    f">>>\tPer-launch # GPU kernel dispatches ({backend}): {aggregate_stats.num_dispatches / timing_args.iter}"
                )
                print(
                    f">>>\tPer-launch GPU mean time ({backend}): {aggregate_stats.mean}us"
                )

            for stat in csv_stats:
                csv_file.write(f"{aggregate_stats._asdict()[stat]},")

        csv_file.write("\n")


SUPPORTED_AGGREGATE_STATS = ["mean", "num_dispatches"]


def get_aggregate_stats(
    csv_stats: list[str], results: ZoneStatsSummary, iter: int
) -> ZoneStats:
    if len(results.keys()) == 1:
        zone_stats = list(results.values())[0]
        return zone_stats
    ret = {}
    for stat in csv_stats:
        if stat not in SUPPORTED_AGGREGATE_STATS:
            ret[stat] = "unsupported stat calculation"
        elif stat == "mean":
            ret[stat] = _get_mean_gpu_time_per_launch(results, iter)
        elif stat == "num_dispatches":
            ret[stat] = _get_total_num_dispatches(results)
    return ZoneStats(**ret)


def run(
    func: Callable,
    timing_args: argparse.Namespace,
    output_num_bytes: int,
    per_device_args: Sequence[tuple[torch.Tensor, ...]],
    devices: Sequence[torch.device],
    verbose: bool,
) -> profile | None:
    """Distributes `iter`-many applications of `func` to `per_device_args`. If
    timing is requested, returns a torch profiler object that can be inspected
    to recover time-related information."""
    # HIP backend caches allocations by default and can OOM if not explicitly cleared.
    for device in devices:
        get_device_from_torch(device).hal_device.allocator.trim()
    # Reset torch.compile caches to avoid hitting re-compile limits.
    torch.compiler.reset()

    num_devices = len(per_device_args)
    # This is a rough threshold: Mi300x 192 GB memory divided by 2.
    mem_bytes_threshold = 96 * (10**9)
    iter_thresh = int(mem_bytes_threshold // output_num_bytes)
    assert (
        iter_thresh > 1 or not timing_args.time
    ), "Cannot reliably profile if cleanup is needed after every step."

    # Cleanup is performed after every iter_thresh steps, including the
    # initialization step and the cleanup steps themselves:
    #   num_cleanups = (iter // num_devices + num_cleanups + 1) // iter_thresh
    # Solving which leads to the form below.
    num_cleanups = (timing_args.iter // num_devices + 1) // (iter_thresh - 1)

    # The total number of iterations includes cleanups and the initial
    # initialization operation that are not profiled, so the profiler records
    # the expected number of iterations. When not profiling, just run as many
    # times as requested.
    total_num_iters = (
        timing_args.iter + (num_cleanups + 1) * num_devices
        if timing_args.time
        else timing_args.iter
    )

    def needs_cleanup(step: int) -> bool:
        per_device_step = step // num_devices
        return (per_device_step + 1) % iter_thresh == 0

    def schedule_fn(step: int) -> torch.profiler.ProfilerAction:
        """Scheduling function for the profiler. Ensures it doesn't capture the
        first iteration and the cleanup iterations where additional overhead may
        happen."""
        # Skip fist run on each device.
        if step < num_devices:
            return torch.profiler.ProfilerAction.NONE

        # Skip the step on which cleanup happens.
        if needs_cleanup(step):
            return torch.profiler.ProfilerAction.NONE

        # Save the results at the last iteration.
        if step == total_num_iters:
            return torch.profiler.ProfilerAction.RECORD_AND_SAVE
        return torch.profiler.ProfilerAction.RECORD

    profile_context = (
        profile(activities=[ProfilerActivity.CUDA], schedule=schedule_fn)
        if timing_args.time
        else nullcontext()
    )

    results: tuple[torch.Tensor, ...] | torch.Tensor | None = None
    with profile_context as prof:
        for iter in range(total_num_iters):
            device_idx = iter % num_devices
            launch_args = per_device_args[device_idx]
            results = func(*launch_args)
            if needs_cleanup(iter):
                print(
                    f">>>\tSynchronizing all devices on iter {iter} and collecting garbage."
                )
                for device in devices:
                    torch.cuda.synchronize(device)
                gc.collect()
            if prof is not None:
                prof.step()

    if results is None:
        results = ()
    if isinstance(results, torch.Tensor):
        results = (results,)
    for i, result in enumerate(results):
        if verbose:
            print(
                f">>>\tresult #{i} shape: {result.shape}; dtype: {result.dtype}; device type: {result.device.type}"
            )
    return prof if timing_args.time else None


def get_torch_compiled_module(signature: OpSignature, backend: str) -> Callable:
    """Returns the module defined by the signature compiled using the specified
    backend."""
    mod = signature.get_nn_module(use_custom=False)
    return torch.compile(mod, dynamic=False, backend=backend)


DEFAULT_BACKEND = "iree_boo_legacy"
BACKEND_TO_FUNC_GENERATOR: dict[str, Callable[[OpSignature], Callable]] = {
    "torch": (lambda signature: signature.get_nn_module(use_custom=False)),
    "inductor": partial(get_torch_compiled_module, backend="inductor"),
    "iree_boo_legacy": get_launchable,
    "iree_boo": partial(get_torch_compiled_module, backend="iree_boo"),
    "iree_boo_experimental": partial(
        get_torch_compiled_module, backend="iree_boo_experimental"
    ),
    "iree_boo_inductor": partial(
        get_torch_compiled_module, backend="iree_boo_inductor"
    ),
}


def _get_total_num_dispatches(results: ZoneStatsSummary) -> int:
    """Returns averaged number of kernel launches per iter. In some cases this could be non-integer."""
    return sum([int(s.num_dispatches) for s in results.values()])


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
    sig: OpSignature, splat_input_value: None | int | float, devices: list[torch.device]
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


def _get_mean_gpu_time_per_launch(results: ZoneStatsSummary, iter: int) -> float:
    total_time = 0.0
    for stats in results.values():
        assert isinstance(stats.mean, float) and isinstance(stats.num_dispatches, int)
        total_time += stats.mean * stats.num_dispatches
    return total_time / iter


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
        max_time = max(times)
        mean_time = statistics.mean(times)
        stddev = statistics.stdev(times) if len(times) > 1 else 0
        count = len(times)
        results[zone_name] = ZoneStats(
            min=min_time,
            max=max_time,
            mean=mean_time,
            stddev=stddev,
            num_dispatches=count,
        )
    return results


def _print_zone_stats(results: ZoneStatsSummary) -> None:
    """Prints a ZoneStatsSummary."""
    for zone_name, stats in results.items():
        s = ">>>\t"
        s += f"{zone_name}\n>>>\t\t"
        decorate_val = lambda val: (
            f"{val:.2f}us" if isinstance(val, float) else str(val)
        )
        s += " ".join(
            [f"{key}={decorate_val(value)}" for key, value in stats._asdict().items()]
        )
        print(s)


if __name__ == "__main__":
    main()
