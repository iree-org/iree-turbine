import argparse
from collections.abc import Callable, Sequence
import shlex
import statistics
import torch


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
    args, extra_cli_args = parser.parse_known_args()

    if args.commands_file:
        with open(args.commands_file) as f:
            mio_file_args = [
                shlex.split(s) for s in f.readlines() if not s.startswith("#")
            ]
    else:
        mio_file_args = [[]]  # use CLI arguments

    runner_parser = argparse.ArgumentParser()
    runner_parser.add_argument("--timing", "-t", type=int, help="Enable timing")
    for file_args in mio_file_args:
        args = file_args + extra_cli_args
        timing_args, runner_args = runner_parser.parse_known_args(args)
        func = lambda: run(runner_args)
        if not timing_args.timing:
            func()
            continue

        zones, func_name = trace_gpu(func)
        dispatch_zone_names = [n for n in zones.keys() if n.startswith(func_name)]
        if len(dispatch_zone_names) == 0:
            print(">>> FAILED TO COLLECT TIMING INFO")
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


def run(cli_args: Sequence[str]):
    # In order to be properly traced only the subprocesses should import
    # 'iree.runtime', so all turbine imports need to be kept local.
    from iree.turbine.kernel.boo.conv_exports import (
        miopen_parser as mio,
        get_launchable,
    )

    print(shlex.join(cli_args))
    parser = mio.get_miopen_parser()
    parser.add_argument(
        "--iter", type=int, help="Number of iterations to run", default=100
    )
    args = parser.parse_args(cli_args)
    sig = mio.get_signature(args)
    conv = get_launchable(sig)

    conv_args = sig.get_sample_conv_args(seed=10, device="cuda")

    result = None
    for _ in range(args.iter):
        result = conv(*conv_args)

    torch.set_printoptions(edgeitems=0)
    print(f">>> {result}")

    return sig.get_func_name()


def trace_gpu(func: Callable[[], str]) -> tuple[dict[str, list[int]], str]:
    """Profile 'func' under Tracy, and return the GPU zone execution times."""
    from multiprocessing import Process, Queue
    import os
    import subprocess
    from subprocess import Popen
    import sys
    from tempfile import TemporaryDirectory

    tracy_port = "44434"
    with TemporaryDirectory() as temp_dir:
        trace_path = f"{temp_dir}/out.trace"
        with Popen(
            ["iree-tracy-capture", "-o", trace_path, "-f", "-p", tracy_port],
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
        ) as tracy:
            queue = Queue()

            def proc_fn():
                os.environ["TRACY_PORT"] = tracy_port
                queue.put(func())

            process = Process(target=proc_fn)
            process.start()
            process.join()
            if process.exitcode != 0:
                sys.exit(process.exitcode)
            result = queue.get()
            try:
                # Tracy will never exit if it fails to connect, so kill the process after some time.
                out, err = tracy.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                tracy.kill()
                out, err = tracy.communicate()
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
