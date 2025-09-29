# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

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

from ..runtime.op_reg import AttrArg, TensorArg

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
class test_optional_returns(CustomOp):
    """A CustomOp testing optional return tensors."""

    @property
    def signature(self) -> str:
        return "test_optional_returns(Tensor a, Tensor b, bool[] mask) -> (Tensor?, Tensor?)"

    def select(self, sel: KernelSelection):
        a_desc = sel.arg_tensor(0)
        b_desc = sel.arg_tensor(1)
        mask_desc = sel.attr_list_bool(2)
        mask = mask_desc.v
        assert (
            isinstance(mask, list) and len(mask) == 2
        ), "Must have two values for mask arg."
        sel.maybe_return_tensor(
            torch.empty(a_desc.t.shape, device="meta") if mask[0] else None
        )
        sel.maybe_return_tensor(
            torch.empty(b_desc.t.shape, device="meta") if mask[1] else None
        )

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        # result type + non-optional args
        a_desc = ksel.arg_descs[0]
        b_desc = ksel.arg_descs[1]
        assert isinstance(a_desc, TensorArg)
        assert isinstance(b_desc, TensorArg)
        rank_a = len(a_desc.t.shape)
        rank_b = len(b_desc.t.shape)
        mask_desc = ksel.arg_descs[2]
        assert isinstance(mask_desc, AttrArg)
        mask = mask_desc.v
        assert isinstance(mask, list)
        mask_str = "mask_" + "_".join([str(m) for m in mask])
        types = [
            a_desc.mlir_type_asm,
            b_desc.mlir_type_asm,
        ]
        names = [r"%a", r"%b"]
        return_vals = ", ".join([name for m, name in zip(mask, names) if m])
        return_types = ", ".join([t for t, m in zip(types, mask) if m])
        return_string = "" if not any(mask) else return_vals + " : " + return_types
        function_name = f"turbine_test_optional_returns_{mask_str}_{rank_a}d_{rank_b}d"
        # Instantiate template
        func_op = _templates.inline_template_function(
            kb,
            "test_optional_returns",
            function_name,
            tensor_type_a=types[0],
            tensor_type_b=types[1],
            mask=mask_str,
            rank_a=rank_a,
            rank_b=rank_b,
            func_return_type=f"({return_types})",
            return_string=return_string,
        )
        kb.yield_results(*impl_helper.call_function(func_op, *kb.arg_bindings[0:2]))  # type: ignore


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
