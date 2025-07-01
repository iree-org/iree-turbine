# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import random
import shlex
import statistics
import copy


def main():
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [... MIOpenDriver command ...] [--commands-file COMMANDS_FILE]",
        description="""
Run a convolution with the IREE runtime. Command line arguments mirror the
arguments to MIOpenDriver.

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
        csv_file.write(shlex.join(driver_args) + ",")
        if not timing_args.time:
            # IMPORTANT: this import must remain here because otherwise it loads
            # IREE runtime (transitively when loading packages through
            # iree.turbine.kernel) that would clash with Tracy options that are
            # needed when running with a timing flag enabled.
            import iree.turbine.kernel.boo.conv_exports.runner as runner

            runner.run(runner_args, args.gpu_id)
            continue

        # Print proof of life.
        print(" ".join(runner_args))

        try:
            zones, func_name = trace_gpu(runner_args, args.gpu_id)
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


TRACY_PORT = str(random.randint(40_000, 50_000))


def trace_gpu(runner_args: str, gpu_id: int) -> tuple[dict[str, list[int]], str]:
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
            environ = copy.deepcopy(os.environ)
            if "BOO_FORCE_TRACY_RUNTIME" in os.environ:
                environ["IREE_PY_RUNTIME"] = "tracy"
            environ["TRACY_PORT"] = TRACY_PORT
            process = subprocess.run(
                ["python", "runner.py", str(gpu_id)] + runner_args,
                env=environ,
                capture_output=True,
                text=True,
            )
            if process.returncode:
                raise RuntimeError(process.stderr)
            result = process.stdout.strip()
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
