#!/bin/bash

set -eu -o errtrace

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
iree_dir="$repo_root/vendor/iree"
build_dir="$repo_root/build"
cache_dir="${cache_dir:-}"

# Setup cache dir.
if [ -z "${cache_dir}" ]; then
  cache_dir="${repo_root}/.shark-build-cache"
  mkdir -p "${cache_dir}"
  cache_dir="$(cd ${cache_dir} && pwd)"
fi
echo "Caching to ${cache_dir}"
mkdir -p "${cache_dir}/ccache"
mkdir -p "${cache_dir}/yum"
mkdir -p "${cache_dir}/pip"

python="$(which python)"
echo "Using python: $python"

export CMAKE_TOOLCHAIN_FILE="$this_dir/linux_default_toolchain.cmake"
export CC=clang
export CXX=clang++
export CCACHE_DIR="${cache_dir}/ccache"
export CCACHE_MAXSIZE="2G"
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Clear ccache stats.
ccache -z

cmake -GNinja -B${build_dir} -S${repo_root} \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE=$python \
  -DPYTHON_EXECUTABLE=$python \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_ENABLE_ASSERTIONS=ON

echo "Building all"
echo "------------"
cmake --build "$build_dir" -- -k 0

echo "Building test deps"
echo "------------------"
cmake --build "$build_dir" --target iree-test-deps -- -k 0

# TODO: Figure out ctest incantation and run tests.

# Show ccache stats.
ccache --show-stats