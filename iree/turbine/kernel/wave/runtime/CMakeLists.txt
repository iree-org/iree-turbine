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
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE torch_cmake_PATH)
list(APPEND CMAKE_PREFIX_PATH ${torch_cmake_PATH})
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

find_package(Torch REQUIRED)
find_package(HIP REQUIRED)
add_definitions(-D__HIP_PLATFORM_AMD__)

# Build the core parts of nanobind once
nanobind_build_library(nanobind SHARED)

# Compile an extension library
add_library(wave_runtime MODULE runtime.cpp)

# .. and link it against the nanobind parts
target_link_libraries(wave_runtime PRIVATE nanobind hip::host ${TORCH_LIBRARIES})
include_directories(${TORCH_INCLUDE_DIRS})

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
