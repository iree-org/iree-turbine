# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
from iree.turbine.kernel.wave.utils import (
    get_default_arch,
)

require_e2e = pytest.mark.require_e2e
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(), reason="Default device is not CDNA3"
)
require_cdna2 = pytest.mark.skipif(
    "gfx90" not in get_default_arch(), reason="Default device is not CDNA2"
)
# Whether to dump the generated MLIR module.
dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

def param_bool(name, short_name, vals=None, *args, **kwargs):
    """Add a boolean parameterization with useful names.
    
    By default the values will be both False and True, but you can pass vals for
    the case where it's helpful to have a test technically be parameterized but
    actually have only one option.
    """
    vals = [False, True] if vals is None else vals
    ids = [short_name if val else f"no_{short_name}" for val in vals]
    return pytest.mark.parametrize(name, vals, ids=ids, *args, **kwargs)
