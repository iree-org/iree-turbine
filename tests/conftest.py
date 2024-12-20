# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--runperf", action="store_true", default=False, help="run performance tests"
    )
    parser.addoption(
        "--dump-perf-files-path",
        action="store",
        default=None,
        help="save performance info into provided directory, filename based on current test name",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "perf_only: performance test, runs only with '--runperf'"
    )
    config.addinivalue_line(
        "markers", "validate_only: validation test, never runs with '--runperf'"
    )


DISTRIBUTE_GPU_TESTS = os.environ.get("WAVE_DISTRIBUTE_GPU_TESTS", None)


def _set_default_device(config):
    if DISTRIBUTE_GPU_TESTS is None:
        return

    if not hasattr(config, "workerinput"):
        return

    workerinput = config.workerinput
    if not workerinput.startswith("gw"):
        return

    device_id = int(workerinput[2:]) % int(DISTRIBUTE_GPU_TESTS)

    import iree.turbine.kernel.wave.utils as utils

    utils.DEFAULT_GPU_DEVICE = device_id


def _has_marker(item, marker):
    return next(item.iter_markers(marker), None) is not None


def pytest_collection_modifyitems(config, items):
    _set_default_device(config)
    run_perf = config.getoption("--runperf")
    for item in items:
        is_validate_only = _has_marker(item, "validate_only")
        is_perf_only = _has_marker(item, "perf_only")
        if run_perf:
            if not is_perf_only or is_validate_only:
                item.add_marker(pytest.mark.skip("skip non-perf test"))
        else:
            if is_perf_only:
                item.add_marker(pytest.mark.skip("skip perf test"))
