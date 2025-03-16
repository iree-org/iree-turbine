# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

from iree.turbine.kernel.boo.conv_exports.conv import ConvSignature
from iree.turbine.kernel.boo.conv_exports.generate import (
    generate_mlir,
    command_to_signature,
)

_sample_pipelines = [None, ["torch-to-iree"], "builtin.module(torch-to-iree)"]
_sample_commands = [
    "conv",
    "convbfp16",
    "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -m conv -g 1 -F 2 -t 1 --iter 100 --in_layout NHWC --out_layout NHWC --fil_layout NHWC",
]
_expected_names = [
    "conv_2d_float32_forward_100x3x32x32_32x3x3x3_1x1s_0x0p_1x1d_1g",
    "conv_2d_bfloat16_forward_100x3x32x32_32x3x3x3_1x1s_0x0p_1x1d_1g",
    "conv_2d_bfloat16_input_backward_16x48x32x96_96x3x1x96_1x1s_2x0p_2x2d_1g",
]


class ConvGeneratorTest(unittest.TestCase):
    def testCustomPipeline(self):
        signature = ConvSignature(input_shape=[1, 2, 16], kernel_shape=[3, 2, 2])
        for pipeline in _sample_pipelines:
            module = generate_mlir(signature, import_pipeline=pipeline)
            self.assertIn(f"@{signature.get_func_name()}(", str(module))

    def testGenerateFromCommand(self):
        for i in range(len(_sample_commands)):
            signature = command_to_signature(_sample_commands[i])
            module = generate_mlir(signature)
            self.assertIn(f"@{_expected_names[i]}", str(module))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
