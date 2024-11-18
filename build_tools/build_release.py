#!/usr/bin/env python
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fetches dependent release artifacts and builds wheels.

See docs/releasing.md for usage.
"""

import argparse
from datetime import date
import json
import os
from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_INFO_FILE = REPO_ROOT / "version_info.json"
WHEEL_DIR = REPO_ROOT / "wheelhouse"

# The platform flags that we will download IREE wheels for. This must match
# the platforms and Python versions we build. If it mismatches or something
# is wrong, this will error. Note that the platform and python-version
# indicates "fetch me a wheel that will install on this combo" vs "fetch me
# a specific wheel".
IREE_PLATFORM_ARGS = [
    # Linux aarch64
    # ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.9"],
    ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.10"],
    ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.11"],
    ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.12"],
    ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.13"],
    # ["--platform", "manylinux_2_28_aarch64", "--python-version", "3.13t"],
    # Linux x86_64
    # ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.9"],
    ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.10"],
    ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.11"],
    ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.12"],
    ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.13"],
    # ["--platform", "manylinux_2_28_x86_64", "--python-version", "3.13t"],
    # MacOS
    ["--platform", "macosx_13_0_universal2", "--python-version", "3.11"],
    ["--platform", "macosx_13_0_universal2", "--python-version", "3.12"],
    ["--platform", "macosx_13_0_universal2", "--python-version", "3.13"],
    # Windows
    ["--platform", "win_amd64", "--python-version", "3.11"],
    ["--platform", "win_amd64", "--python-version", "3.12"],
    ["--platform", "win_amd64", "--python-version", "3.13"],
]


def eval_version(version_spec: str):
    date_stamp = date.today().strftime("%Y%m%d")
    return version_spec.replace("YYYYMMDD", date_stamp)


def write_version_info(args):
    with open(VERSION_INFO_FILE, "rt") as f:
        info_dict = json.load(f)

    # Compute package-version.
    package_version = eval_version(args.package_version)
    if args.package_pre_version:
        package_version += eval_version(args.package_pre_version)
    if args.package_post_version:
        package_version += f".{eval_version(args.package_post_version)}"
    info_dict["package-version"] = package_version

    with open(VERSION_INFO_FILE, "wt") as f:
        json.dump(info_dict, f, indent=2)
        f.write("\n")

    print(f"Updated version_info.json:\n{json.dumps(info_dict, indent=2)}")


def exec(args, env=None):
    args = [str(s) for s in args]
    print(f": Exec: {shlex.join(args)}")
    if env is not None:
        full_env = dict(os.environ)
        full_env.update(env)
    else:
        full_env = None
    subprocess.check_call(args, env=full_env)


def download_requirements(requirements_file, platforms=()):
    args = [
        "pip",
        "download",
        "-d",
        WHEEL_DIR,
    ]
    if platforms:
        args.append("--no-deps")
        for p in platforms:
            args.extend(["--platform", p])
    args += [
        "-f",
        WHEEL_DIR,
        "-r",
        requirements_file,
    ]
    exec(args)


def download_iree_binaries():
    for platform_args in IREE_PLATFORM_ARGS:
        print("Downloading for platform:", platform_args)
        args = [
            "pip",
            "download",
            "-d",
            WHEEL_DIR,
            "--no-deps",
        ]
        args.extend(platform_args)
        args += [
            # Uncomment to allow nightly releases (if not pinned in the file)
            # "-f",
            # "https://iree.dev/pip-release-links.html",
            "-f",
            WHEEL_DIR,
            # Note: could also drop `-ci` here, if coordinating a release
            # across projects and new stable versions of the IREE packages
            # haven't yet been pushed.
            "-r",
            REPO_ROOT / "iree-requirements-ci.txt",
        ]
        exec(args)


def build_wheel(path, env=None):
    exec(
        ["pip", "wheel", "--no-index", "-f", WHEEL_DIR, "-w", WHEEL_DIR, path], env=env
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-version", help="Version to use", required=True)
    parser.add_argument(
        "--package-pre-version",
        help="Pre-release version segment or (YYYYMMDD)",
        default="",
    )
    parser.add_argument(
        "--package-post-version",
        help="Post-release version segment or (YYYYMMDD)",
        default="",
    )
    parser.add_argument(
        "--no-download", help="Disable dep download", action="store_true"
    )
    args = parser.parse_args()

    write_version_info(args)
    WHEEL_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        print("Prefetching all IREE binaries")
        download_iree_binaries()
        print("Prefetching torch CPU")
        download_requirements(REPO_ROOT / "pytorch-cpu-requirements.txt")
        print("Downloading remaining requirements")
        download_requirements(REPO_ROOT / "requirements.txt")

    print("Building iree-turbine")
    build_wheel(REPO_ROOT)


if __name__ == "__main__":
    main()
