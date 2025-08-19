# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest


@pytest.fixture(scope="function", autouse=True)
def seed_torch():
    import torch

    torch.manual_seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e", action="store_true", default=False, help="run e2e tests"
    )
    parser.addoption(
        "--run-expensive-tests",
        action="store_true",
        default=False,
        help="run expensive tests",
    )
    parser.addoption(
        "--runperf", action="store_true", default=False, help="run performance tests"
    )
    parser.addoption(
        "--dump-perf-files-path",
        action="store",
        default=None,
        help="save performance info into provided directory, filename based on current test name",
    )
    parser.addoption(
        "--gpu-distribute",
        type=int,
        default=0,
        help="Distribute over N gpu devices when running with pytest-xdist",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "require_e2e: e2e test, runs with '--run-e2e'")
    config.addinivalue_line(
        "markers", "expensive_test: expensive test, runs with '--run-expensive-tests'"
    )
    config.addinivalue_line(
        "markers", "perf_only: performance test, runs only with '--runperf'"
    )
    config.addinivalue_line(
        "markers", "validate_only: validation test, never runs with '--runperf'"
    )


def _get_worker_id(config):
    """
    Returns the worker id for the current worker if running with pytest-xdist.
    None if pytest-xdist is not installed or not running in parallel.
    """
    # Extract the worker id using internal pytest APIs.
    if not hasattr(config, "workerinput"):
        return None

    # workerid has format 'gw0', 'gw1', etc.
    worker_id = config.workerinput["workerid"]
    if not worker_id.startswith("gw"):
        return None

    # skip the 'gw' prefix.
    return int(worker_id[2:])


def _has_marker(item, marker):
    return next(item.iter_markers(marker), None) is not None


def pytest_collection_modifyitems(config, items):
    run_e2e = config.getoption("--run-e2e")
    run_expensive = config.getoption("--run-expensive-tests")
    run_perf = config.getoption("--runperf")
    for item in items:
        if _has_marker(item, "require_e2e") and not run_e2e:
            item.add_marker(pytest.mark.skip("e2e tests are disabled"))

        if _has_marker(item, "expensive_test") and not run_expensive:
            item.add_marker(pytest.mark.skip("expensive tests are disabled"))

        is_validate_only = _has_marker(item, "validate_only")
        is_perf_only = _has_marker(item, "perf_only")
        if run_perf:
            if not is_perf_only or is_validate_only:
                item.add_marker(pytest.mark.skip("skip non-perf test"))
        else:
            if is_perf_only:
                item.add_marker(pytest.mark.skip("skip perf test"))
