# Copyright 2024 Advanced Micro Devices, inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
This pass will add a specified dictionary as an iree.reflection attribute to a module's public function(s).
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import re

from iree.turbine.support.ir_imports import *

from ..rewriter import *
from iree.compiler.ir import Context, DictAttr


def value_to_attr(value):
    val_type = type(value).__name__
    match val_type:
        case "str":
            return StringAttr.get(value)
        case _:
            return StringAttr.get(str(value))


class AddMetadataPass(Pass):
    def __init__(
        self,
        mlir_module: Module,
        inp_metadata: dict,
        func_name: str,
    ):
        super().__init__(mlir_module.operation)
        self.mlir_module = mlir_module
        self.inp_metadata = inp_metadata
        self.func_name = func_name
        self.context = self.mlir_module.context

    def run(self):
        def parse_metadata_dict(metadata_dict: dict) -> DictAttr:
            with self.context:
                for key, value in metadata_dict.items():
                    metadata_dict[key] = value_to_attr(value)
                metadata_dict = DictAttr.get(metadata_dict)
                return metadata_dict

        metadata_dict_attr = parse_metadata_dict(self.inp_metadata)
        for func_op in self.funcs:
            ir_func_symbol = SymbolTable.get_symbol_name(func_op.op)
            ir_func_symbol_name = StringAttr(ir_func_symbol).value
            if ir_func_symbol_name == self.func_name:
                func_op.op.attributes["iree.reflection"] = metadata_dict_attr
        return self.mlir_module


if __name__ == "__main__":
    pass_main(AddMetadataPass)
