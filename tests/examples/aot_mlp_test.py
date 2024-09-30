# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from pathlib import Path
import sys
import subprocess
import unittest

REPO_DIR = Path(__file__).resolve().parent.parent.parent


def _run(local_path: str):
    path = REPO_DIR / local_path
    subprocess.check_call([sys.executable, str(path)])


class AOTMLPTest(unittest.TestCase):
    def testMLPExportSimple(self):
        _run("examples/aot_mlp/mlp_export_simple.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
