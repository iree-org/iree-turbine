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

# Set up environment.
python3.11 -m venv "${WHEELHOUSE_DIR}"/test.venv
source "${WHEELHOUSE_DIR}"/test.venv/bin/activate

# Install wheels
# --no-index is required so that we don't pick up different versions from pypi
pip install --no-index -f "${WHEELHOUSE_DIR}" iree-turbine[testing]
pip install --no-index -f "${WHEELHOUSE_DIR}" torchvision
pip freeze

# Run tests
pytest -n 4 "${REPO_ROOT}"
