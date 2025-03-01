#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scripts grabs the X.Y.Z[.dev]` version identifier from `version.json`
# and writes the corresponding `X.Y.ZrcYYYYMMDD` version identifier to
# `version_local.json`.

from datetime import datetime
from packaging.version import Version
from pathlib import Path
import argparse
import json
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--write-json", action="store_true")
parser.add_argument("--version-suffix", action="store", type=str)

release_type = parser.add_mutually_exclusive_group(required=True)
release_type.add_argument("-stable", "--stable-release", action="store_true")
release_type.add_argument("-rc", "--nightly-release", action="store_true")
release_type.add_argument("-dev", "--development-release", action="store_true")

args = parser.parse_args()

REPO_ROOT = Path(__file__).parent.parent
VERSION_FILE_PATH = REPO_ROOT / "version.json"
VERSION_LOCAL_FILE_PATH = REPO_ROOT / "version_local.json"


def load_version_from_file(version_file):
    with open(version_file, "rt") as f:
        return json.load(f)


def write_version_to_file(version_file, version):
    with open(version_file, "w") as f:
        json.dump({"package-version": version}, f, indent=2)
        f.write("\n")


version_info = load_version_from_file(VERSION_FILE_PATH)
package_version = version_info.get("package-version")
current_version = Version(package_version).base_version

if args.nightly_release:
    current_version += "rc" + datetime.today().strftime("%Y%m%d")
elif args.development_release:
    current_version += (
        ".dev0+"
        + subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )

if args.version_suffix:
    current_version += args.version_suffix

if args.write_json:
    write_version_to_file(VERSION_LOCAL_FILE_PATH, current_version)

print(current_version)
