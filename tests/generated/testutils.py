import time
import types
import os
import re
import sys
from functools import partial
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading
import signal
import platform
import resource
import logging
from tqdm import *

from stats import Stats, ErrorAggregatorDict
from evaluate import evaluate_importer

log = logging.getLogger("turbine-test")


def call_with_timeout(fn, args, kwargs=None, timeout=10):
    kwargs = kwargs or {}
    parent_conn, child_conn = multiprocessing.Pipe()
    start = time.time()
    proc = multiprocessing.Process(
        target=call_with_timeout_subproc, args=(fn, args, kwargs, child_conn)
    )
    proc.start()
    while proc.is_alive():
        if parent_conn.poll(1):
            result = parent_conn.recv()
            proc.join()
            return result
        if time.time() - start > timeout:
            os.kill(
                proc.pid, signal.SIGINT
            )  # maybe generate a stack trace for debugging
            time.sleep(1)
            proc.terminate()
            proc.join(10)
            raise TimeoutError(f"took longer than {timeout} seconds")

    proc.join()
    if proc.exitcode == 0:
        return parent_conn.recv()
    else:
        raise OSError(f"exitcode should be 0, got {proc.exitcode}")


def call_with_timeout_subproc(fn, args, kwargs, return_pipe):
    # use_rlimit = (
    #     os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024 ** 3 < 1000
    #     if platform.system() == "Linux"
    #     else True
    # )
    # if use_rlimit:
    #     _, hard = resource.getrlimit(resource.RLIMIT_AS)
    #     # resource.setrlimit(resource.RLIMIT_AS, (int(os.environ.get("RLIMIT_AS_GB", 10)) * 1024 ** 3, hard))
    try:
        result = fn(*args, *kwargs)
        return_pipe.send(result)
    except Exception:
        log.exception("Error from subprocess")
        sys.exit(1)


def subproc_wrapper(path: str, fn: callable, timeout: int = 900):
    """
    A wrapper around call_with_timeout() adding a temp dir and error handling.

    :param path: path to code to test
    :param fn: function to run in subprocess
    :param timeout: seconds to wait
    :return: errors, stats
    """
    file = os.path.basename(path).split("/")[-1]
    test_identifier = re.sub(r"\.py$", "", file)

    log.info(f"Running {path}")
    try:
        return call_with_timeout(fn, [path], {}, timeout=timeout)
    except TimeoutError as e:
        return ErrorAggregatorDict.single(str(e), test_identifier), Stats(
            {"TIMEOUT": 1}
        )
    except OSError as e:
        return ErrorAggregatorDict.single(str(e), test_identifier), Stats(
            {"CRASHED": 1}
        )


def import_file(path):
    """
    :param path: to a *.py file
    :return: a python module
    """
    module = types.ModuleType(re.findall(r"test_[^.]+", path)[0])
    sys.modules[module.__name__] = module
    exec(
        compile(open(path).read(), filename=path, mode="exec"),
        module.__dict__,
        module.__dict__,
    )
    if not hasattr(module, "TESTCASES"):
        module.TESTCASES = []

    return module


def evaluate_pyfile_subproc(path: str, args, eval_fn=evaluate_importer):
    """
    Evaluate/test all the TESTCASES in path.

    :param path: *.py file to test
    :return: errors, stats
    """
    errors = ErrorAggregatorDict()
    stats = Stats()
    module = import_file(path)

    if not module.TESTCASES:
        log.info(f"Skipping empty module: {module.__name__}")
        stats["SKIPPED"] += 1
        return errors, stats

    index = -1
    for nn_cls, get_init_args, get_forward_args, compiles in module.TESTCASES:
        index += 1
        stats["TOTAL"] += 1

        if args.filter and args.filter not in nn_cls.__name__:
            stats["SKIPPED"] += 1
            continue

        if args.skips and f"{nn_cls.__name__}" in args.skips:
            stats["SKIPPED"] += 1
            continue

        # nn.module doesn't have `forward` function(e.g, has __call__ instead).
        # dynamo doesn't plan to support it yet.
        if nn_cls.forward.__name__ == "_forward_unimplemented":
            stats["NO_FWD"] += 1
            continue

        repro = f"{nn_cls.__name__} # pytest {path} -k test_{index:03d}"
        test_identifier = f"{module.__name__}__{index:03d}"
        eval_args = [nn_cls, get_init_args, get_forward_args, test_identifier]

        try:
            err_dict = eval_fn(*eval_args)
            if err_dict and len(err_dict):
                log.info(f"{test_identifier} - FAIL")
                errors.update(err_dict)
                stats["FAILED"] += 1
            else:
                log.info(f"{test_identifier} - PASS")
                stats["PASSED"] += 1
        except Exception as e:
            log.info(f"{test_identifier} - FAIL (Exception)")
            errors.insert(str(e), test_identifier)

    return errors, stats


def evaluate_all(
    args, tests_dir: str = "./generated", offset: int = 0, limit: int = None, jobs=4
):
    """
    Generate a paritybench score, main entrypoint for this module.

    :param tests_dir: directory containing paritybench testcases
    :param limit: optional maximum number of files to process
    :param fn: inner function to run the tests
    :param jobs: how many processes to run at once
    """
    feval = partial(evaluate_pyfile_subproc, args=args)
    fn = partial(subproc_wrapper, fn=feval)
    start = time.time()
    stats = Stats()
    errors = ErrorAggregatorDict()
    testfiles = [
        os.path.join(tests_dir, f)
        for f in os.listdir(tests_dir)
        if re.search(r"test_.*[.]py$", f)
    ]
    testfiles.sort()

    if limit:
        testfiles = testfiles[offset : offset + limit]

    with tqdm(total=len(testfiles)) as pbar:
        if args.sequential:
            for file in testfiles:
                errors_part, stats_part = fn(path=file)
                errors.update(errors_part)
                stats.update(stats_part)
                pbar.update()
        else:
            pool = ThreadPool(jobs)
            for errors_part, stats_part in pool.imap_unordered(fn, testfiles):
                errors.update(errors_part)
                stats.update(stats_part)
                pbar.update()
            pool.close()

    errors.print_report()
    log.info(f"Total time: {time.time() - start:02f} s")
    log.info(stats)
