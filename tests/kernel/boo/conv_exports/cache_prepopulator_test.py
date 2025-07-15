# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from iree.turbine.kernel.boo.runtime import set_cache_dir, clear_cache


class CachePopulatorTest(unittest.TestCase):
    def testPopulator(self):
        with TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_cache_dir(cache_dir=cache_dir)

            from iree.turbine.kernel.boo.driver.preload import CachePopulator
            from iree.turbine.kernel.boo.conv_exports.miopen_parser import ConvParser
            from iree.turbine.kernel.boo.conv_exports.conv import ConvSignature

            commands = [
                "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
                "convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
            ]

            pop = CachePopulator(
                commands=commands, parser_cls=ConvParser, signature_cls=ConvSignature
            )

            pop.run()

            self.assertTrue(
                len(pop.signatures) == 2, "Number of signatures should be 2."
            )
            self.assertTrue(pop.commands is None, "Commands should be cleared.")
            self.assertTrue(pop.commands_file is None, "Commands_file should be None.")

            for sig in pop.signatures:
                name = sig.get_func_name()
                sub_dir = set_cache_dir() / name
                self.assertTrue(
                    sub_dir.is_dir(),
                    f"CachePopulator must generate sub directory {sub_dir}.",
                )
                mlir_file = sub_dir / f"{name}.mlir"
                self.assertTrue(
                    mlir_file.is_file(), f"Expected mlir file at {mlir_file}"
                )
                count = len([f for f in sub_dir.glob("*.vmfb")])
                self.assertGreater(count, 0, "Expected at least one vmfb.")

            clear_cache()
            self.assertFalse(cache_dir.is_dir(), f"Expected cache dir to be cleared.")


if __name__ == "__main__":
    unittest.main()
