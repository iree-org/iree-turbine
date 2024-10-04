# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


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


def pytest_collection_modifyitems(config, items):
    run_perf = config.getoption("--runperf")
    for item in items:
        is_perf_only = next(item.iter_markers("perf_only"), None) is not None
        if run_perf:
            if not is_perf_only:
                item.add_marker(pytest.mark.skip("skip non-perf test"))
        else:
            if is_perf_only:
                item.add_marker(pytest.mark.skip("skip perf test"))
