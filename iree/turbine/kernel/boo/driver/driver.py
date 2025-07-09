# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import gc
import argparse
from typing import Callable, Sequence
import os
import random
import shlex
import statistics
import torch

# NOTE: must not import anything form iree.turbine here because that *may*
# cause IREE runtime to be loaded, which affects Tracy's ability to collect
# kernel trace data. This has some negative implications, like the impossibility
# to easily fetch the list of available ops or parser options or other
# per-operation aspects and other unnecessary coupling, but it is caused, at
# least in part, by IREE shipping two runtimes (clean and tracy-enabled) via PIP
# and not doing that in local builds. See
# https://github.com/iree-org/iree-turbine/pull/987#discussion_r2175341728
# for more context.


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [... MIOpenDriver command ...] [--commands-file COMMANDS_FILE]",
        description="""
Run a kernel with the IREE runtime. Command line arguments mirror the
arguments to MIOpenDriver.

Currently supports convolution and layernorm.

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
            zones, func_name = trace_gpu(func)
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
            # Convert from nanoseconds to microseconds
            times = [t / 1000 for t in zones[zone_name]]
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
            csv_file.write(f"{min(zones[dispatch_zone_names[0]]) / 1000:.2f}\n")


def run(cli_args: Sequence[str], gpu_id: int):
    # In order to be properly traced only the subprocesses should import
    # 'iree.runtime', so all turbine imports need to be kept local.

    from iree.turbine.kernel.boo.exports.parser import OpCLIParser

    def dispatch(cli_args: Sequence[str]) -> type[OpCLIParser]:
        if any("conv" in x for x in cli_args):
            from iree.turbine.kernel.boo.conv_exports.miopen_parser import ConvParser

            return ConvParser
        if any("layernorm" in x for x in cli_args):
            from iree.turbine.kernel.boo.layer_norm_exports.miopen_parser import (
                LayerNormParser,
            )

            return LayerNormParser
        raise ValueError("unsupported operation kind in " + shlex.join(cli_args))

    from iree.turbine.kernel.boo.driver.launch import get_launchable

    print(shlex.join(cli_args))
    parser_cls = dispatch(cli_args)
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
    results = results or ()
    if isinstance(results, torch.Tensor):
        results = (results,)
    for i, result in enumerate(results):
        print(
            f">>> result #{i} shape: {result.shape}; dtype: {result.dtype}; device type: {result.device.type}"
        )

    return sig.get_func_name()


TRACY_PORT = str(random.randint(40_000, 50_000))


def trace_gpu(func: Callable[[], str]) -> tuple[dict[str, list[int]], str]:
    """Profile 'func' under Tracy, and return the GPU zone execution times."""
    from multiprocessing import Process, Queue
    import os
    import subprocess
    from subprocess import Popen
    import sys
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp_dir:
        trace_path = f"{temp_dir}/out.trace"
        with Popen(
            ["iree-tracy-capture", "-o", trace_path, "-f", "-p", TRACY_PORT],
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        ) as tracy:
            queue = Queue()

            def proc_fn():
                os.environ["TRACY_PORT"] = TRACY_PORT
                try:
                    queue.put(func())
                except Exception as exc:
                    queue.put(str(exc))
                    raise

            process = Process(target=proc_fn)
            process.start()
            process.join()
            result = queue.get_nowait()
            if process.exitcode != 0:
                raise ValueError(result)
            try:
                # Tracy will never exit if it fails to connect, so kill the process after some time.
                out, err = tracy.communicate(timeout=5)
            except subprocess.TimeoutExpired as e:
                tracy.kill()
                raise ValueError("Tracy failed to connect.") from e
        if tracy.returncode:
            raise ValueError(f"Tracy failed:\n{out}\n{err}")

        csvexport = subprocess.run(
            ["tracy-csvexport", "--gpu", trace_path],
            capture_output=True,
            check=True,
            text=True,
        )

    import csv

    reader = csv.reader(csvexport.stdout.splitlines())
    header = next(reader)
    column = {name: idx for idx, name in enumerate(header)}

    zones: dict[str, list[int]] = {}
    for row in reader:
        name = row[column["name"]]
        time = int(row[column["GPU execution time"]])
        zones.setdefault(name, []).append(time)

    return zones, result


if __name__ == "__main__":
    main()
