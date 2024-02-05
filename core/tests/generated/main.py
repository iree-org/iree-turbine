import os
import sys
import argparse

from testutils import evaluate_all

# magic: makes importing this module work in the testsuite
import torch._inductor.config

import logging

log = logging.getLogger("turbine-test")
logging.basicConfig(level=logging.INFO)

ENV_FILE = "JITPARITYBENCH_PATH.txt"


def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=4,
        help="Number of threads in our threadpool, ignored if --sequential is set",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Pick files starting from this offset. Together with --limit, we can run through all files in multiple separate runs",
    )
    parser.add_argument("--limit", "-l", type=int, help="only run the first N files")
    parser.add_argument(
        "--filter", "-f", "-k", help="only run module containing given name"
    )
    parser.add_argument("--skips", type=str)
    parser.add_argument(
        "--tests-dir",
        default=None,
        help="jit-paritybench location (i.e. /path/to/pytorch-jit-paritybench)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Set to run tests sequentially without threading, this can help resolve hanging or long runtimes due to low memory",
    )
    # parser.add_argument("--device", default="cuda", type=str, help="evaluate modules using cuda or cpu") # excluded for now as we only have turbine-cpu, can use this later

    args = parser.parse_args(raw_args)
    return args


def write_path(path: str):
    with open(ENV_FILE, "w") as f:
        f.write(path)


def read_path() -> str:
    with open(ENV_FILE, "r") as f:
        path = f.read()
    return path


if __name__ == "__main__":
    args = get_args()

    if args.tests_dir is not None:
        pb = args.tests_dir
        write_path(pb)  # store this path for next time
        log.info(f"Using test directory from CLI: {pb}")
    elif os.path.exists(ENV_FILE):
        pb = read_path()
        log.info(f"Using test directory from {ENV_FILE}: {pb}")
    else:
        raise RuntimeError(
            f"Must either pass 'tests-dir' or set {ENV_FILE} in order to run tests"
        )

    # enables finding necessary modules in jit-paritybench
    pb_gen = pb + "/generated"
    sys.path.append(pb)
    sys.path.append(pb_gen)

    evaluate_all(args, pb_gen, offset=args.offset, limit=args.limit, jobs=args.jobs)
