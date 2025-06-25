# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.15...3.27)
project(wave_runtime)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Link against torch, using the libraries and include dirs in site-packages.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.__path__[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_PATH)

# Add all directories in /opt/rocm/lib/cmake to CMAKE_PREFIX_PATH
file(GLOB subdirectories "/opt/rocm/lib/cmake/*")
foreach(subdir ${subdirectories})
  if(IS_DIRECTORY ${subdir})
    list(APPEND CMAKE_PREFIX_PATH "${subdir}")
  endif()
endforeach()

include_directories(${torch_PATH}/include ${torch_PATH}/include/torch/csrc/api/include/)
link_directories(${torch_PATH}/lib)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_cxx_abi)
set(TORCH_CXX_FLAGS -D_GLIBCXX_USE_CXX11_ABI=${torch_cxx_abi})

find_package(hip REQUIRED)
add_definitions(-D__HIP_PLATFORM_AMD__ -DUSE_ROCM)

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)

# Compile an extension library
add_library(wave_runtime MODULE runtime.cpp)

# .. and link it against the nanobind parts
set(TORCH_LIBRARIES ${torch_PATH}/lib/libtorch.so ${torch_PATH}/lib/libc10.so)
target_link_libraries(wave_runtime PRIVATE nanobind hip::host ${TORCH_LIBRARIES})
set_target_properties(wave_runtime PROPERTIES LINK_WHAT_YOU_USE TRUE)

# .. enable size optimizations
nanobind_opt_size(wave_runtime)

# .. enable link time optimization
nanobind_lto(wave_runtime)

# .. set the default symbol visibility to 'hidden'
nanobind_set_visibility(wave_runtime)

# .. strip unneeded symbols and debug info from the binary (only active in release builds)
nanobind_strip(wave_runtime)

# .. disable the stack protector
nanobind_disable_stack_protector(wave_runtime)

# .. set the Python extension suffix
nanobind_extension(wave_runtime)

# .. set important compilation flags
nanobind_compile_options(wave_runtime PRIVATE ${TORCH_CXX_FLAGS})

# .. set important linker flags
nanobind_link_options(wave_runtime)

install(TARGETS wave_runtime DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
