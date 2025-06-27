# Copyright 2023 Advanced Micro Devices, Inc.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import distutils.command.build
from pathlib import Path
import subprocess
import sys
import shutil

from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_ROOT = THIS_DIR

VERSION_FILE = os.path.join(REPO_ROOT, "version.json")
VERSION_FILE_LOCAL = os.path.join(REPO_ROOT, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def get_package_version():
    """Load and return the package version from version files."""
    try:
        version_info = load_version_info(VERSION_FILE_LOCAL)
    except FileNotFoundError:
        print("version_local.json not found. Default to dev build")
        version_info = load_version_info(VERSION_FILE)

    package_version = version_info["package-version"]
    print(f"Using PACKAGE_VERSION: '{package_version}'")
    return package_version


def load_requirement_pins(requirements_file: str):
    """Load version pins from requirements file."""
    with open(Path(THIS_DIR) / requirements_file, "rt") as f:
        lines = f.readlines()
    pin_pairs = [line.strip().split("==") for line in lines if "==" in line]
    return dict(pin_pairs)


def get_version_spec(dep: str, requirement_pins: dict):
    """Get version specification for a dependency."""
    if dep in requirement_pins:
        return f">={requirement_pins[dep]}"
    return ""


def check_nanobind_available():
    """Check if nanobind is available in the current Python environment."""
    try:
        import nanobind

        return True
    except ImportError:
        return False


def check_torch_cuda_available():
    """Check if PyTorch has CUDA support."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def setup_cmake_environment():
    """Setup CMake environment with system Python."""
    python_executable = sys.executable
    cmake_env = os.environ.copy()
    cmake_args = []

    print(f"Using Python: {python_executable}")

    return python_executable, cmake_env, cmake_args


# CMake extension support for wave runtime
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Check dependencies before attempting CMake build
        nanobind_available = check_nanobind_available()
        torch_cuda_available = check_torch_cuda_available()

        if not nanobind_available:
            print("Warning: nanobind not found. Wave runtime will be disabled.")
            print("To enable wave runtime, install nanobind: pip install nanobind")

        if not torch_cuda_available:
            print(
                "Warning: PyTorch CUDA support not available. Wave runtime will be disabled."
            )
            print("To enable wave runtime, install PyTorch with CUDA support.")

        # Ensure CMake is available
        cmake_cmd = "cmake"

        try:
            subprocess.check_output([cmake_cmd, "--version"])
            print(f"Successfully verified cmake at: {cmake_cmd}")
        except OSError as e:
            print(f"Failed to run cmake: {e}")
            raise RuntimeError(
                "CMake must be installed system-wide to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
                + "\n\nPlease install CMake using your system package manager:\n"
                + "  Ubuntu/Debian: sudo apt-get install cmake\n"
                + "  macOS: brew install cmake\n"
                + "  Windows: Download from https://cmake.org/download/\n"
                + "  Or install via pip: pip install cmake"
            )

        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        # Get extension directory
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Setup CMake environment
        python_executable, cmake_env, cmake_args = setup_cmake_environment()

        # Configure CMake with dependency flags
        cmake_args.extend(
            [
                f"-DPython_EXECUTABLE={python_executable}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
                f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
                f"-DNANOBIND_AVAILABLE={'ON' if nanobind_available else 'OFF'}",
                f"-DTORCH_CUDA_AVAILABLE={'ON' if torch_cuda_available else 'OFF'}",
            ]
        )

        print(f"CMake args: {cmake_args}")
        print(f"CMake source dir: {ext.sourcedir}")
        print(f"CMake build dir: {build_dir}")
        subprocess.check_call(
            [cmake_cmd, ext.sourcedir, *cmake_args], cwd=build_dir, env=cmake_env
        )

        # Build CMake project
        subprocess.check_call([cmake_cmd, "--build", "."], cwd=build_dir)


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


def get_extension_files():
    """Get list of compiled extension files to include in the package."""
    extension_files = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(current_dir):
        if file.endswith((".so", ".dll", ".dylib", ".pyd")):
            extension_files.append(file)

    return extension_files


def main():
    """Main setup function."""
    # Load version and requirements
    package_version = get_package_version()
    requirement_pins = load_requirement_pins("requirements.txt")

    # Load README
    with open(os.path.join(REPO_ROOT, "README.md"), "rt") as f:
        readme = f.read()

    # Find packages
    packages = find_namespace_packages(
        include=[
            "iree.turbine",
            "iree.turbine.*",
        ],
    )
    print("Found packages:", packages)

    # Get extension files
    extension_files = get_extension_files()
    print(f"Extension files to include: {extension_files}")

    setup(
        name="iree-turbine",
        version=f"{package_version}",
        author="IREE Authors",
        author_email="iree-technical-discussion@lists.lfaidata.foundation",
        description="IREE Turbine Machine Learning Deployment Tools",
        long_description=readme,
        long_description_content_type="text/markdown",
        license="Apache-2.0",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
        ],
        project_urls={
            "homepage": "https://iree.dev/",
            "repository": "https://github.com/iree-org/iree-turbine/",
            "documentation": "https://iree-turbine.readthedocs.io/en/latest/",
        },
        packages=packages,
        include_package_data=True,
        package_data={
            "iree.turbine": [
                "ops/templates/*.mlir",
                "kernel/boo/conv_exports/tuning_specs.mlir",
                # Include compiled extensions
                "*.so",
                "*.dll",
                "*.dylib",
                "*.pyd",
            ],
        },
        data_files=[
            # Include compiled extensions in the root package
            ("", extension_files),
        ],
        ext_modules=[
            CMakeExtension(
                "wave_runtime", sourcedir="iree/turbine/kernel/wave/runtime"
            ),
        ],
        entry_points={
            "torch_dynamo_backends": [
                "turbine_cpu = iree.turbine.dynamo.backends.base:backend",
                "iree_turbine = iree.turbine.dynamo.backends.base:backend",
            ],
        },
        install_requires=[
            f"numpy{get_version_spec('numpy', requirement_pins)}",
            f"iree-base-compiler{get_version_spec('iree-base-compiler', requirement_pins)}",
            f"iree-base-runtime{get_version_spec('iree-base-runtime', requirement_pins)}",
            f"Jinja2{get_version_spec('Jinja2', requirement_pins)}",
            f"ml_dtypes{get_version_spec('ml_dtypes', requirement_pins)}",
            f"typing_extensions{get_version_spec('typing_extensions', requirement_pins)}",
        ],
        extras_require={
            "testing": [
                f"pytest{get_version_spec('pytest', requirement_pins)}",
                f"pytest-xdist{get_version_spec('pytest-xdist', requirement_pins)}",
                f"parameterized{get_version_spec('parameterized', requirement_pins)}",
            ],
        },
        cmdclass={
            "build": BuildCommand,
            "build_ext": CMakeBuild,
        },
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
