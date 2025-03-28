import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from iree.turbine.kernel.boo.conv_exports.launch import set_boo_cache, clear_cache_dir
from iree.turbine.support.logging import runtime_logger as logger


class CachePopulatorTest(unittest.TestCase):
    def testPopulator(self):
        with TemporaryDirectory() as td:
            cache_dir = Path(td)
            set_boo_cache(cache_dir=cache_dir)

            from iree.turbine.kernel.boo.conv_exports.launch import CACHE_BASE_DIR

            self.assertTrue(
                CACHE_BASE_DIR == cache_dir,
                f"Mismatch in cache dirs. Set {cache_dir=} but got {CACHE_BASE_DIR=}.",
            )

            from iree.turbine.kernel.boo.conv_exports import CachePopulator

            commands = [
                "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
                "convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
            ]

            pop = CachePopulator(commands=commands)

            pop.run()

            self.assertTrue(
                len(pop.signatures) == 2, "Number of signatures should be 2."
            )
            self.assertTrue(pop.commands is None, "Commands should be cleared.")
            self.assertTrue(pop.commands_file is None, "Commands_file should be None.")

            for sig in pop.signatures:
                name = sig.get_func_name()
                sub_dir = CACHE_BASE_DIR / name
                self.assertTrue(
                    sub_dir.is_dir(),
                    f"CachePopulator must generate sub directory {sub_dir}.",
                )
                mlir_file = sub_dir / f"{name}.mlir"
                self.assertTrue(
                    mlir_file.is_file(), f"Expected mlir file at {mlir_file}"
                )

            clear_cache_dir()
            self.assertFalse(cache_dir.is_dir(), f"Expected cache dir to be cleared.")


if __name__ == "__main__":
    unittest.main()
