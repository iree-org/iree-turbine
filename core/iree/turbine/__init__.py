"""
The turbine package provides development tools for deploying PyTorch 2 machine
learning models to cloud and edge devices.
"""

# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: This redirection layer exists while we are migrating from the
# shark_turbine top-level package name to iree.turbine. It exports the
# public API but not the internal details. In a future switch, all code
# will be directly located here and the redirect will be done in the
# shark_turbine namespace.

from shark_turbine import aot
from shark_turbine import dynamo
from shark_turbine import kernel
from shark_turbine import ops
from shark_turbine import runtime
