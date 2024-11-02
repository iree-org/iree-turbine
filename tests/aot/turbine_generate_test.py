# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import io

from iree.build import *


def test_example_builder(tmp_path):
    from iree.turbine.aot.testing import example_builder

    iree_build_main(
        example_builder,
        args=(
            f"--output-dir={tmp_path}",
            "--iree-hal-target-device=cpu",
            "--iree-llvmcpu-target-cpu=host",
        ),
    )

    # Should have compiled three outputs.
    for output_name in [
        "bin/pipe/stage0_cpu-host.vmfb",
        "bin/pipe/stage1_cpu-host.vmfb",
        "bin/pipe/stage2_cpu-host.vmfb",
    ]:
        output_path = tmp_path / output_name
        assert output_path.exists()

    # Should have generated with a dynamic batch and two fixed batch sizes.
    for gen_name, contains_str in [
        ("genfiles/pipe/import_stage0.mlir", "!torch.vtensor<[?,64],f32>"),
        ("genfiles/pipe/import_stage1.mlir", "!torch.vtensor<[10,64],f32>"),
        ("genfiles/pipe/import_stage2.mlir", "!torch.vtensor<[20,64],f32>"),
    ]:
        gen_path = tmp_path / gen_name
        contents = gen_path.read_text()
        assert contains_str in contents
