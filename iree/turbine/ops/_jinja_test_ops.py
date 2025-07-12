# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.ir_imports import (
    RankedTensorType,
    IrType,
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

LIBRARY = def_library("_turbine_jinja_test")
_templates = impl_helper.JinjaTemplateLoader(__name__)


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
        function_name = f"turbine_test_add_jinja_{rtt.rank}d_{str(rtt.element_type)}"
        func_op = _templates.inline_template_function(
            kb,
            "test_add_jinja",
            function_name,
            rank=rtt.rank,
            element_type=str(rtt.element_type),
            tensor_type=str(rtt),
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))


@CustomOp.register(library=LIBRARY)
class test_linear_trailing_optional(CustomOp):
    signature = "test_linear_trailing_optional(Tensor t1, Tensor t2, Tensor? t3=None) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        t3_desc = ksel.arg_optional_tensor(2)
        if t3_desc:
            t3_desc.specialize_all_dims()
            ksel.variant = "biased"

        result_desc = ksel.return_new_tensor(
            [t1_desc.t.shape[0], t2_desc.t.shape[1]], t1_desc.t.dtype
        )
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        # result type + non-optional args
        t1_desc = ksel.arg_descs[0]
        t2_desc = ksel.arg_descs[1]
        res_desc = ksel.result_descs[0]

        # Create MLIR type for result
        result_type = IrType.parse(res_desc.mlir_type_asm)

        # Instantiate template
        if ksel.variant == "biased":
            function_name = f"turbine_test_linear_trailing_optional_{result_type.rank}d_{str(result_type.element_type)}"
            t3_desc = ksel.arg_descs[2]
            function_name += "_biased"

            func_op = _templates.inline_template_function(
                kb,
                "test_linear_trailing_optional_biased",
                function_name,
                rank=result_type.rank,
                element_type=result_type.element_type,
                tensor_A_type=t1_desc.mlir_type_asm,  # type: ignore
                tensor_B_type=t2_desc.mlir_type_asm,  # type: ignore
                tensor_bias_type=t3_desc.mlir_type_asm,  # type: ignore
                tensor_Result_type=res_desc.mlir_type_asm,
            )
        else:
            function_name = f"turbine_test_linear_{result_type.rank}d_{str(result_type.element_type)}"
            func_op = _templates.inline_template_function(
                kb,
                "test_linear",
                function_name,
                rank=result_type.rank,
                element_type=result_type.element_type,
                tensor_A_type=t1_desc.mlir_type_asm,  # type: ignore
                tensor_B_type=t2_desc.mlir_type_asm,  # type: ignore
                tensor_Result_type=res_desc.mlir_type_asm,
            )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))


@CustomOp.register(library=LIBRARY)
class test_linear_middle_optional(CustomOp):
    """A CustomOp version of linear, but with the bias passed as an optional second argument rather than trailing argument."""

    signature = (
        "test_linear_middle_optional(Tensor t1, Tensor? t2, Tensor t3) -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_optional_tensor(1)
        if t2_desc:
            t2_desc.specialize_all_dims()
            ksel.variant = "biased"
        t3_desc = ksel.arg_tensor(2)
        t3_desc.specialize_all_dims()

        result_desc = ksel.return_new_tensor(
            [t1_desc.t.shape[0], t3_desc.t.shape[1]], t1_desc.t.dtype
        )
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        # result type + non-optional args
        t1_desc = ksel.arg_descs[0]
        t3_desc = ksel.arg_descs[2]
        res_desc = ksel.result_descs[0]

        # Create MLIR type for result
        result_type = IrType.parse(res_desc.mlir_type_asm)

        # Instantiate template
        if ksel.variant == "biased":
            t2_desc = ksel.arg_descs[1]
            function_name = f"turbine_test_linear_middle_optional_{result_type.rank}d_{str(result_type.element_type)}_biased"

            func_op = _templates.inline_template_function(
                kb,
                "test_linear_middle_optional_biased",
                function_name,
                rank=result_type.rank,
                element_type=result_type.element_type,
                tensor_A_type=t1_desc.mlir_type_asm,  # type: ignore
                tensor_B_type=t3_desc.mlir_type_asm,  # type: ignore
                tensor_bias_type=t2_desc.mlir_type_asm,  # type: ignore
                tensor_Result_type=res_desc.mlir_type_asm,
            )
        else:
            function_name = f"turbine_test_linear_{result_type.rank}d_{str(result_type.element_type)}"
            func_op = _templates.inline_template_function(
                kb,
                "test_linear",
                function_name,
                rank=result_type.rank,
                element_type=result_type.element_type,
                tensor_A_type=t1_desc.mlir_type_asm,  # type: ignore
                tensor_B_type=t3_desc.mlir_type_asm,  # type: ignore
                tensor_Result_type=res_desc.mlir_type_asm,
            )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings))
