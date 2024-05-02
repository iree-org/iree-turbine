# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.ir_imports import (
    RankedTensorType,
)

from ..runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
    def_library,
    impl_helper,
)

__all__ = [
    "trace",
]

LIBRARY = def_library("_turbine_str_format_test")
_templates = impl_helper.StrFormatTemplateLoader(__name__)


@CustomOp.register(library=LIBRARY)
class test_add(CustomOp):
    signature = "test_add(Tensor t1, Tensor t2) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor(list(t1_desc.t.shape), t1_desc.t.dtype)
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        result_type = kb.arg_bindings[0].type  # type: ignore
        rtt = RankedTensorType(result_type)
        function_name = (
            f"turbine_test_add_strformat_{rtt.rank}d_{str(rtt.element_type)}"
        )
        func_op = _templates.inline_template_function(
            kb,
            "test_add_strformat",
            function_name,
            rank=rtt.rank,
            element_type=str(rtt.element_type),
            tensor_type=str(rtt),
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))


@CustomOp.register(library=LIBRARY)
class syntax_error(CustomOp):
    signature = "syntax_error(Tensor t1) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor(list(t1_desc.t.shape), t1_desc.t.dtype)
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        function_name = "syntax_error"
        func_op = _templates.inline_template_function(
            kb,
            "test_syntax_error",
            function_name,
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))
