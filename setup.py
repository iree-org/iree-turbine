# Copyright 2023 Advanced Micro Devices, Inc.
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import distutils.command.build
import subprocess
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_ROOT = THIS_DIR


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        # Create build directory
        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        # Get extension directory
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Configure CMake
        cmake_args = [
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={'Debug' if self.debug else 'Release'}",
        ]
        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_dir)

        # Build CMake project
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)


VERSION_FILE = os.path.join(REPO_ROOT, "version.json")
VERSION_FILE_LOCAL = os.path.join(REPO_ROOT, "version_local.json")


def load_version_info(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


try:
    version_info = load_version_info(VERSION_FILE_LOCAL)
except FileNotFoundError:
    print("version_local.json not found. Default to dev build")
    version_info = load_version_info(VERSION_FILE)

PACKAGE_VERSION = version_info["package-version"]
print(f"Using PACKAGE_VERSION: '{PACKAGE_VERSION}'")

with open(os.path.join(REPO_ROOT, "README.md"), "rt") as f:
    README = f.read()

packages = find_namespace_packages(
    include=[
        "iree.turbine",
        "iree.turbine.*",
    ],
)

print("Found packages:", packages)

# Lookup version pins from requirements files.
requirement_pins = {}


def load_requirement_pins(requirements_file: str):
    with open(Path(THIS_DIR) / requirements_file, "rt") as f:
        lines = f.readlines()
    pin_pairs = [line.strip().split("==") for line in lines if "==" in line]
    requirement_pins.update(dict(pin_pairs))


load_requirement_pins("requirements.txt")


def get_version_spec(dep: str):
    if dep in requirement_pins:
        return f">={requirement_pins[dep]}"
    else:
        return ""


# Override build command so that we can build into _python_build
# instead of the default "build". This avoids collisions with
# typical CMake incantations, which can produce all kinds of
# hilarity (like including the contents of the build/lib directory).
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.build_base = "_python_build"


setup(
    name="iree-turbine",
    version=f"{PACKAGE_VERSION}",
    author="IREE Authors",
    author_email="iree-technical-discussion@lists.lfaidata.foundation",
    description="IREE Turbine Machine Learning Deployment Tools",
    long_description=README,
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
        ],  # Include MLIR templates
    },
    entry_points={
        "torch_dynamo_backends": [
            "turbine_cpu = iree.turbine.dynamo.backends.base:backend",
            "iree_turbine = iree.turbine.dynamo.backends.base:backend",
            "iree_boo = iree.turbine.dynamo.backends.boo:default_backend",
            "iree_boo_inductor = iree.turbine.dynamo.backends.boo:inductor_backend",
        ],
    },
    install_requires=[
        f"numpy{get_version_spec('numpy')}",
        f"iree-base-compiler{get_version_spec('iree-base-compiler')}",
        f"iree-base-runtime{get_version_spec('iree-base-runtime')}",
        f"Jinja2{get_version_spec('Jinja2')}",
        f"ml_dtypes{get_version_spec('ml_dtypes')}",
        f"typing_extensions{get_version_spec('typing_extensions')}",
    ],
    extras_require={
        "testing": [
            f"pytest{get_version_spec('pytest')}",
            f"pytest-xdist{get_version_spec('pytest-xdist')}",
            f"parameterized{get_version_spec('parameterized')}",
        ],
    },
    cmdclass={"build": BuildCommand, "build_ext": CMakeBuild},
    ext_modules=[CMakeExtension("wave_runtime", "iree/turbine/kernel/wave/runtime")],
    zip_safe=False,
)
