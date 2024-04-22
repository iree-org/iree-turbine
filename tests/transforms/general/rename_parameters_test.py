# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging
import unittest

from iree.compiler.ir import (
    Context,
    Operation,
)

from shark_turbine.transforms import rewriter
from shark_turbine.transforms.general import rename_parameters

SIMPLE_GLOBALS_ASM = r"""
module {
    util.global private @_params.classifier.default {noinline} = #stream.parameter.named<"default"> : tensor<30xf32>
    util.global private @_params.classifier.weight {noinline} = #stream.parameter.named<"foo"::"WEIGHT"> : tensor<30x20xf32>
    util.global private @_params.classifier.bias {noinline} = #stream.parameter.named<"foo"::"params.classifier.bias"> : tensor<30xf32>
    util.global private @_params.classifier.other {noinline} = dense<0.0> : tensor<30xf32>
    util.global private @_uninitialized {noinline} : tensor<30xf32>
}
"""


class RenameTest(unittest.TestCase):
    def testBasic(self):
        with Context() as context:
            module_op = Operation.parse(SIMPLE_GLOBALS_ASM)
            rename_parameters.RenameParametersPass(
                module_op,
                rename_map={
                    "WEIGHT": "weight",
                    ("foo", "params.classifier.bias"): ("bar", "BIAS"),
                },
                rename_callback=lambda scope, name: ("XXX", "YYY")
                if name == "default"
                else None,
            ).run()
            module_asm = str(module_op)
            print(module_asm)
            self.assertIn(
                '@_params.classifier.default {noinline} = #stream.parameter.named<"XXX"::"YYY"> : tensor<30xf32>',
                module_asm,
            )
            self.assertIn(
                '@_params.classifier.weight {noinline} = #stream.parameter.named<"foo"::"weight"> : tensor<30x20xf32>',
                module_asm,
            )
            self.assertIn(
                '@_params.classifier.bias {noinline} = #stream.parameter.named<"bar"::"BIAS"> : tensor<30xf32>',
                module_asm,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
