# Copyright 2023 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os
import distutils.command.build
from pathlib import Path

from setuptools import find_namespace_packages, setup

THIS_DIR = os.path.realpath(os.path.dirname(__file__))
REPO_DIR = os.path.dirname(THIS_DIR)
VERSION_INFO_FILE = os.path.join(REPO_DIR, "version_info.json")

# Transitional as we migrate from shark-turbine -> iree-turbine.
TURBINE_PACKAGE_NAME = os.getenv("TURBINE_PACKAGE_NAME", "shark-turbine")

with open(
    os.path.join(
        REPO_DIR,
        "README.md",
    ),
    "rt",
) as f:
    README = f.read()


def load_version_info():
    with open(VERSION_INFO_FILE, "rt") as f:
        return json.load(f)


version_info = load_version_info()
PACKAGE_VERSION = version_info["core-version"]

packages = find_namespace_packages(
    include=[
        "iree.turbine",
        "iree.turbine.*",
        "shark_turbine",
        "shark_turbine.*",
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


load_requirement_pins("iree-requirements.txt")
load_requirement_pins("misc-requirements.txt")
load_requirement_pins("pytorch-cpu-requirements.txt")


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
    name=f"{TURBINE_PACKAGE_NAME}",
    version=f"{PACKAGE_VERSION}",
    author="SHARK Authors",
    author_email="stella@nod.ai",
    description="SHARK Turbine Machine Learning Deployment Tools",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/nod-ai/SHARK-Turbine",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=packages,
    entry_points={
        "torch_dynamo_backends": [
            "turbine_cpu = shark_turbine.dynamo.backends.cpu:backend",
        ],
    },
    install_requires=[
        f"numpy{get_version_spec('numpy')}",
        f"iree-compiler{get_version_spec('iree-compiler')}",
        f"iree-runtime{get_version_spec('iree-runtime')}",
        # Use the [torch-cpu-nightly] spec to get a more recent/specific version.
        # Note that during the transition to torch 2.3.0 we technically support
        # back to torch 2.1, which is why we pin here in this way. However,
        # the CI tests on 2.3.
        "torch>=2.1.0",
    ],
    extras_require={
        "torch-cpu-nightly": [f"torch{get_version_spec('torch')}"],
        "onnx": [
            f"onnx{get_version_spec('onnx')}",
        ],
        "testing": [
            f"onnx{get_version_spec('onnx')}",
            f"pytest{get_version_spec('pytest')}",
            f"pytest-xdist{get_version_spec('pytest-xdist')}",
        ],
    },
    cmdclass={"build": BuildCommand},
)
