# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel.boo.layer_norm_exports.miopen_parser import LayerNormParser
from iree.turbine.kernel.boo.driver.numerics import run_numerics

if __name__ == "__main__":
    run_numerics(LayerNormParser, use_custom=True)
