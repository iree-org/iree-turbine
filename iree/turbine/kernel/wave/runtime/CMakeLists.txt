# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.19...3.27)
project(wave_runtime)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)

# Compile an extension library
add_library(wave_runtime MODULE runtime.cpp)

# .. and link it against the nanobind parts
target_link_libraries(wave_runtime PRIVATE nanobind)

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

# .. set important linker flags
nanobind_link_options(wave_runtime)

install(TARGETS wave_runtime DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
