# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from iree.turbine.kernel.boo.runtime import set_cache_dir, clear_cache
import pytest
import torch


def _marked_xfail(*args):
    return pytest.param(
        *args,
        marks=pytest.mark.xfail(
            condition=not torch.cuda.is_available(),
            reason="CPU layernorm/gemm compile failure",
        ),
    )


@pytest.mark.parametrize("add_gpu_only", (_marked_xfail(True), False))
def testPopulator(add_gpu_only: bool, boo_cache_dir: Path):
    from iree.turbine.kernel.boo.driver.preload import CachePopulator

    commands = [
        "convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
        "convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
    ]
    if add_gpu_only:
        commands.append("layernorm --input 2x3x4x5")
        commands.append("gemm --a_w 32 --a_h 64 --b_w 128")

    pop = CachePopulator(commands=commands)
    pop.run()

    expected = 4 if add_gpu_only else 2
    assert (
        len(pop.signatures) == expected
    ), f"Number of signatures should be {expected}."
    assert pop.commands is None, "Commands should be cleared."
    assert pop.commands_file is None, "Commands_file should be None."

    for sig in pop.signatures:
        name = sig.func_name
        sub_dir = set_cache_dir() / name
        assert (
            sub_dir.is_dir()
        ), f"CachePopulator must generate sub directory {sub_dir}."
        mlir_file = sub_dir / f"{name}.mlir"
        assert mlir_file.is_file(), f"Expected mlir file at {mlir_file}"
        count = len([f for f in sub_dir.glob("*.vmfb")])
        assert count > 0, "Expected at least one vmfb."

    clear_cache()
    assert not boo_cache_dir.is_dir(), f"Expected cache dir to be cleared."
