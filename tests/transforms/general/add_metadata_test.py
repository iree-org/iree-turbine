# Copyright 2024 Advanced Micro Devices, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging
import unittest

from iree.compiler.ir import Context, Operation, Module

from iree.turbine.transforms.general import add_metadata

SIMPLE_FUNC_ASM = r"""
func.func @list_func(%arg0 : !util.list<!util.variant>) -> !util.list<!util.variant> {
  return %arg0 : !util.list<!util.variant>
}
"""


class MetadataTest(unittest.TestCase):
    def testBasic(self):
        metadata_dict = {
            "test_data_str": "test_data_str_value",
            "test_data_int": 42,
            "test_data_dict": {"test_data_dict_key": "test_data_dict_value"},
            "test_data_list": ["test_data_list_value"],
            "test_data_float": 3.14159,
            "test_data_bool": True,
            "test_data_tuple": (1,),
        }
        with Context() as context:
            module = Module.parse(SIMPLE_FUNC_ASM)
            module_op = add_metadata.AddMetadataPass(
                module.operation,
                metadata_dict,
                "list_func",
            ).run()
            module_asm = str(module_op)
            print(module_asm)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
