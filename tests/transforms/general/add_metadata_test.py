# Copyright 2022 The IREE Authors
# Copyright 2024 Advanced Micro Devices, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import logging
import unittest

from iree.compiler.ir import Context, Operation, Module

from shark_turbine.transforms.general import add_metadata

SIMPLE_FUNC_ASM = r"""
func.func @list_func(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant> {
  return %arg0 : !iree_input.list<!iree_input.variant>
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
