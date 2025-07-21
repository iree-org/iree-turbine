# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel.boo.driver.preload import preload
from iree.turbine.kernel.boo.gemm_exports.gemm import GEMMSignature
from iree.turbine.kernel.boo.gemm_exports.miopen_parser import GEMMParser

if __name__ == "__main__":
    preload(GEMMParser, GEMMSignature)
