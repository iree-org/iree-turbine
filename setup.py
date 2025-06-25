# Copyright 2023 Advanced Micro Devices, Inc.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License 2.0 with LLVM Exceptions.
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


def find_cmake_executable():
    """Find the best available cmake executable."""
    # Try to find system cmake
    system_cmake = shutil.which(
        "cmake", path="/usr/bin:/usr/local/bin:/opt/homebrew/bin"
    )
    if system_cmake:
        print(f"Found system cmake at: {system_cmake}")
        return system_cmake

    # Fallback to PATH cmake
    cmake_path = shutil.which("cmake")
    if cmake_path:
        print(f"Found cmake in PATH at: {cmake_path}")
        return cmake_path

    print("Warning: cmake not found in PATH")
    return "cmake"


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
        # Verify that extensions were built
        self.verify_extensions()

    def build_cmake(self, ext):
        # Ensure CMake is available
        cmake_cmd = find_cmake_executable()

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

        # Configure CMake
        cmake_args.extend(
            [
                f"-DPython_EXECUTABLE={python_executable}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
                f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
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

        # Copy the built extension to the source directory for packaging
        self.copy_extension_to_source(ext, extdir)

    def copy_extension_to_source(self, ext, extdir):
        """Copy built extension to source directory for packaging."""
        # Find the built extension file
        ext_name = ext.name
        source_dir = ext.sourcedir

        # Look for the built extension in the build directory
        for file in os.listdir(extdir):
            if file.startswith(ext_name) and file.endswith(
                (".so", ".dll", ".dylib", ".pyd")
            ):
                source_file = os.path.join(extdir, file)
                target_file = os.path.join(source_dir, file)
                print(f"Copying {source_file} to {target_file}")
                shutil.copy2(source_file, target_file)
                break

    def verify_extensions(self):
        """Verify that all extensions were built successfully."""
        for ext in self.extensions:
            source_dir = ext.sourcedir
            ext_name = ext.name

            # Check if extension exists in source directory
            found = False
            for file in os.listdir(source_dir):
                if file.startswith(ext_name) and file.endswith(
                    (".so", ".dll", ".dylib", ".pyd")
                ):
                    print(f"Extension {ext_name} built successfully: {file}")
                    found = True
                    break

            if not found:
                print(f"Warning: Extension {ext_name} not found in {source_dir}")
                print(f"This may cause issues when packaging for PyPI")


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

    # Add wave runtime extension
    if os.path.exists("iree/turbine/kernel/wave/runtime"):
        # Look for compiled extensions in the wave runtime directory
        wave_runtime_dir = "iree/turbine/kernel/wave/runtime"
        for file in os.listdir(wave_runtime_dir):
            if file.endswith((".so", ".dll", ".dylib", ".pyd")):
                extension_files.append(f"{wave_runtime_dir}/{file}")

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
