"""
The turbine package provides development tools for deploying PyTorch 2 machine
learning models to cloud and edge devices.
"""

# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util

msg = """No module named 'torch'. Follow https://pytorch.org/get-started/locally/#start-locally to install 'torch'.
For example, on Linux to install with CPU support run:
  pip3 install torch --index-url https://download.pytorch.org/whl/cpu
"""

if spec := importlib.util.find_spec("torch") is None:
    raise ModuleNotFoundError(msg)
