# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
from iree.turbine.kernel.wave.utils.run_utils import (
    get_default_arch,
)

require_e2e = pytest.mark.require_e2e
expensive_test = pytest.mark.expensive_test
require_cdna2 = pytest.mark.skipif(
    "gfx90" not in get_default_arch(),
    reason="Default architecture is not CDNA2, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(),
    reason="Default architecture is not CDNA3, default architecture is '{}'".format(
        get_default_arch()
    ),
)
require_cdna4 = pytest.mark.skipif(
    "gfx95" not in get_default_arch(),
    reason="Default architecture is not CDNA4, default architecture is '{}'".format(
        get_default_arch()
    ),
)
# Whether to dump the generated MLIR module.
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
expensive_test_param = lambda *a: pytest.param(*a, marks=pytest.mark.expensive_test)


def param_bool(name, shortname=None, values=None):
    shortname = shortname or name
    values = values or [False, True]
    ids = [f"{shortname}" if v else f"no_{shortname}" for v in values]
    return pytest.mark.parametrize(name, [pytest.param(v) for v in values], ids=ids)
