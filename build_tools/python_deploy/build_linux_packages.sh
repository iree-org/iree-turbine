#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# build_linux_packages.sh
# One stop build of SHARK-Turbins Python packages for Linux. The Linux build is
# complicated because it has to be done via a docker container that has
# an LTS glibc version, all Python packages and other deps.
# This script handles all of those details.
#
# Usage:
# Build everything (all packages, all python versions):
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Build specific Python versions and packages to custom directory:
#   override_python_versions="cp38-cp38 cp39-cp39" \
#   packages="iree-runtime iree-runtime-instrumented" \
#   output_dir="/tmp/wheelhouse" \
#   ./build_tools/python_deploy/build_linux_packages.sh
#
# Valid Python versions match a subdirectory under /opt/python in the docker
# image. Typically:
#   cp38-cp38 cp39-cp39 cp310-cp310
#
# Valid packages:
#   iree-runtime
#   iree-runtime-instrumented
#   iree-compiler
#
# Note that this script is meant to be run on CI and it will pollute both the
# output directory and in-tree build/ directories (under runtime/ and
# compiler/) with docker created, root owned builds. Sorry - there is
# no good way around it.
#
# It can be run on a workstation but recommend using a git worktree dedicated
# to packaging to avoid stomping on development artifacts.
set -xeu -o errtrace

# Function to find the directory the ".git" directory is in.
# We do this instead of using git directly because `git` may complain about
# operating in a directory owned by another user.
function find_git_dir_parent() {
  curr_dir="${PWD}"

  # Loop until we reach the root directory
  while [ "${curr_dir}" != "/" ]; do
    # Check if there is a ".git" directory in the current directory
    if [ -d "${curr_dir}/.git" ]; then
      # Return the path to the directory containing the ".git" directory
      echo "${curr_dir}"
      return
    fi

    # Move up one directory
    curr_dir="$(dirname "${curr_dir}")"
  done

  # If we reach the root directory and there is no ".git" directory, return an empty string
  echo ""
}

if [[ "$(uname -m)" == "x86_64" ]]; then
  default_docker_image="quay.io/pypa/manylinux_2_28_x86_64@sha256:4d2e4da308dc8f17418a22ce9396a3cfb930abf16c9c6d5f565cea7316bd1766" # Pinned on 2023-07-18
elif [[ "$(uname -m)" == "aarch64" ]]; then
  default_docker_image="quay.io/pypa/manylinux_2_28_aarch64"
else
  echo "Unknown machine format: Cannot select docker image"
  exit 1
fi

this_dir="$(cd $(dirname $0) && pwd)"
script_name="$(basename $0)"
repo_root=$(cd "${this_dir}" && find_git_dir_parent)
manylinux_docker_image="${manylinux_docker_image:-${default_docker_image}}"
python_versions="${override_python_versions:-}"
output_dir="${output_dir:-${this_dir}/wheelhouse}"
cache_dir="${cache_dir:-}"
packages="${packages:-shark-turbine iree-runtime}"
package_suffix="${package_suffix:-}"

function run_on_host() {
  echo "Running on host"
  echo "Launching docker image ${manylinux_docker_image}"

  # Detect default host python version and use that if not overriden.
  if [ -z "${python_versions}" ]; then
    python_versions="$(python -c "import sys;v=f'cp{sys.version_info.major}{sys.version_info.minor}';print(f'{v}-{v}')")"
    if [ -z "${python_versions}" ]; then
      echo "Python host version autodetect failed. Specify override_python_versions."
      exit 1
    else
      echo "Auto-detected host python version ${python_versions}"
    fi
  fi

  # Canonicalize paths.
  mkdir -p "${output_dir}"
  output_dir="$(cd "${output_dir}" && pwd)"
  echo "Outputting to ${output_dir}"
  mkdir -p "${output_dir}"

  # Make sure that the parent directory of the repository is mapped
  # into Docker (since we store deps side-by-side.)
  super_project_root="$(dirname ${repo_root})"
  echo "Mapping parent dir into docker ${super_project_root}"

  # Setup cache dir.
  if [ -z "${cache_dir}" ]; then
    cache_dir="${super_project_root}/.shark-build-cache"
    mkdir -p "${cache_dir}"
    cache_dir="$(cd ${cache_dir} && pwd)"
  fi
  echo "Caching to ${cache_dir}"
  mkdir -p "${cache_dir}/ccache"
  mkdir -p "${cache_dir}/yum"
  mkdir -p "${cache_dir}/pip"

  # Note that /root/.cache/pip is pretty standard but can be printed
  # in the container with `/opt/python/cp310-cp310/bin/pip cache dir`
  docker run --rm \
    -v "${super_project_root}:${super_project_root}" \
    -v "${output_dir}:${output_dir}" \
    -v "${cache_dir}:${cache_dir}" \
    -v "${cache_dir}/yum:/var/cache/yum" \
    -v "${cache_dir}/pip:/root/.cache/pip" \
    -e __MANYLINUX_BUILD_WHEELS_IN_DOCKER=1 \
    -e "override_python_versions=${python_versions}" \
    -e "packages=${packages}" \
    -e "package_suffix=${package_suffix}" \
    -e "output_dir=${output_dir}" \
    -e "cache_dir=${cache_dir}" \
    "${manylinux_docker_image}" \
    -- ${this_dir}/${script_name}

  echo "******************** BUILD COMPLETE ********************"
  echo "Generated binaries:"
  ls -l "${output_dir}"
}

function run_in_docker() {
  echo "Running in docker"
  echo "Marking git safe.directory"
  git config --global --add safe.directory '*'

  echo "Using python versions: ${python_versions}"
  local orig_path="${PATH}"

  # Configure native builds to use ccache.
  export CCACHE_DIR="${cache_dir}/ccache"
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache

  # Configure yum to keep its cache.
  echo "keepcache = 1" >> /etc/yum.conf
  echo 'cachedir=/var/cache/yum/$basearch/$releasever' >> /etc/yum.conf

  # Build phase.
  for package in ${packages}; do
    echo "******************** BUILDING PACKAGE ${package} ********************"

    # Python version independent packages only build on the first Python version.
    built_shark_turbine=false

    for python_version in ${python_versions}; do
      python_dir="/opt/python/${python_version}"
      if ! [ -x "${python_dir}/bin/python" ]; then
        echo "ERROR: Could not find python: ${python_dir} (skipping)"
        continue
      fi
      export PATH="${python_dir}/bin:${orig_path}"
      echo ":::: Python version $(python --version)"
      # replace dashes with underscores
      package_suffix="${package_suffix//-/_}"
      case "${package}" in
        shark-turbine)
          if ! $built_shark_turbine; then
            clean_wheels "shark_turbine${package_suffix}" "${python_version}"
            build_shark_turbine
            built_shark_turbine=true
          else
            echo "Not building shark-turbine (already built)"
          fi
          ;;
        iree-runtime)
          install_native_build_deps
          clean_wheels "iree_runtime${package_suffix}" "${python_version}"
          build_iree_runtime
          run_audit_wheel "iree_runtime${package_suffix}" "${python_version}"
          ;;
        iree-runtime-instrumented)
          install_native_build_deps
          clean_wheels "iree_runtime_instrumented${package_suffix}" "${python_version}"
          build_iree_runtime_instrumented
          run_audit_wheel "iree_runtime_instrumented${package_suffix}" "${python_version}"
          ;;
        iree-compiler)
          install_native_build_deps        
          clean_wheels "iree_compiler${package_suffix}" "${python_version}"
          build_iree_compiler
          run_audit_wheel "iree_compiler${package_suffix}" "${python_version}"
          ;;
        *)
          echo "Unrecognized package '${package}'"
          exit 1
          ;;
      esac
    done
  done
}

function build_wheel() {
  python -m pip wheel --disable-pip-version-check -v -w "${output_dir}" "${repo_root}/$@"
}

function build_shark_turbine() {
  build_wheel .
}

function build_iree_runtime() {
  IREE_HAL_DRIVER_CUDA=$(uname -m | awk '{print ($1 == "x86_64") ? "ON" : "OFF"}') \
  build_wheel runtime/
}

function build_iree_runtime_instrumented() {
  IREE_HAL_DRIVER_CUDA=$(uname -m | awk '{print ($1 == "x86_64") ? "ON" : "OFF"}') \
  IREE_BUILD_TRACY=ON IREE_ENABLE_RUNTIME_TRACING=ON \
  IREE_RUNTIME_CUSTOM_PACKAGE_SUFFIX="-instrumented" \
  build_wheel runtime/
}

function build_iree_compiler() {
  IREE_TARGET_BACKEND_CUDA=$(uname -m | awk '{print ($1 == "x86_64") ? "ON" : "OFF"}') \
  build_wheel compiler/
}

function run_audit_wheel() {
  local wheel_basename="$1"
  local python_version="$2"
  # Force wildcard expansion here
  generic_wheel="$(echo "${output_dir}/${wheel_basename}-"*"-${python_version}-linux_$(uname -m).whl")"
  ls "${generic_wheel}"
  echo ":::: Auditwheel ${generic_wheel}"
  auditwheel repair -w "${output_dir}" "${generic_wheel}"
  rm -v "${generic_wheel}"
}

function clean_wheels() {
  local wheel_basename="$1"
  local python_version="$2"
  echo ":::: Clean wheels ${wheel_basename} ${python_version}"
  rm -f -v "${output_dir}/${wheel_basename}-"*"-${python_version}-"*".whl"
}

function install_native_build_deps() {
  touch_file="/tmp/native_build_deps_installed"
  if [ -f "${touch_file}" ]; then
    echo ":::: Not installing native build deps (already installed)"
  else
    echo ":::: Instlaling native build deps"
  fi

  # Get the output of uname -m
  uname_m=$(uname -m)

  # Check if the output is aarch64
  if [[ "$uname_m" == "aarch64" ]] || [[ "$uname_m" == "x86_64" ]]; then
    echo "The architecture is aarch64 and we use manylinux 2_28 so install deps"
    needed_packages="ccache capstone-devel tbb-devel libzstd-devel"
    if ! (yum --config /etc/yum.conf install -C -y epel-release $needed_packages); then
      echo "Could not install from yum cache... doing it the slow way."
      yum --config /etc/yum.conf install -y epel-release
      yum --config /etc/yum.conf update -y
      # Required for Tracy
      yum --config /etc/yum.conf install -y $needed_packages
    fi
  else
    echo "The architecture is unknown. Exiting"
    exit 1
  fi

  touch "${touch_file}"
}


# Trampoline to the docker container if running on the host.
if [ -z "${__MANYLINUX_BUILD_WHEELS_IN_DOCKER-}" ]; then
  run_on_host "$@"
else
  run_in_docker "$@"
fi
