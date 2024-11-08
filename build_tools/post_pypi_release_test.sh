#!/bin/bash
# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

THIS_DIR="$(cd $(dirname $0) && pwd)"
REPO_ROOT="$(cd ${THIS_DIR?}/.. && pwd)"
WHEELHOUSE_DIR="${REPO_ROOT?}/wheelhouse"

# Use same environment from build_release, but uninstall the local wheels
source "${WHEELHOUSE_DIR}"/test.venv/bin/activate
pip uninstall -y shark-turbine iree-turbine iree-compiler iree-runtime

# Install from pypi now that latest is released
pip install iree-turbine

# Run tests
pytest -n 4 "${REPO_ROOT}"
