#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This script promotes Python packages from nightly releases to PyPI.
#
# Prerequisites:
#   * For deploying to PyPI, you will need to have credentials set up. See
#     https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives
#     (Googlers can also access the shared releasing account "google-iree-pypi-deploy"
#     at http://go/iree-pypi-password)
#   * Install requirements, e.g. in a Python virtual environment (venv):
#     `pip install -r requirements-packaging.txt`
#   * Choose a release candidate to promote from
#     https://github.com/iree-org/iree-turbine/releases/tag/dev-wheels
#
# Usage:
#   python -m venv .venv
#   source .venv/bin/activate
#   pip install -r ./requirements-packaging.txt
#   ./pypi_deploy.sh 3.1.0rc20241204

set -euo pipefail

RELEASE="$1"

SCRIPT_DIR="$(dirname -- "$( readlink -f -- "$0"; )")";
TMPDIR="$(mktemp --directory --tmpdir iree_turbine_pypi_wheels.XXXXX)"
ASSETS_PAGE="https://github.com/iree-org/iree-turbine/releases/expanded_assets/dev-wheels"

function download_wheels() {
  echo ""
  echo "Downloading wheels for '${RELEASE}'..."

  python -m pip download iree-turbine==${RELEASE} --no-deps -f ${ASSETS_PAGE}

  echo ""
  echo "Downloaded wheels:"
  ls
}

function edit_release_versions() {
  echo ""
  echo "Editing release versions..."
  for file in *
  do
    ${SCRIPT_DIR}/promote_whl_from_rc_to_final.py ${file} --delete-old-wheel
  done

  echo "Edited wheels:"
  ls
}

function upload_wheels() {
  # TODO: list packages that would be uploaded, pause, prompt to continue
  echo ""
  echo "Uploading wheels:"
  ls
  twine upload --verbose *
}

function main() {
  echo "Changing into ${TMPDIR}"
  cd "${TMPDIR}"
  # TODO: check_requirements (using pip)

  download_wheels
  edit_release_versions
  upload_wheels
}

main
