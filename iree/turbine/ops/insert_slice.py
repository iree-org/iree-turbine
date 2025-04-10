# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple, no_type_check
import torch

from ..support.ir_imports import (
    RankedTensorType,
    IrType,
)

from ..runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
    impl_helper,
    TensorArg,
)

from ..support.logging import aot_logger as logger

_templates = impl_helper.JinjaTemplateLoader(__name__)


def unpack_tensor_type(tensor_type: IrType) -> Tuple[str, str, IrType]:
    """Unpacks a RankedTensorType into components usually needed for templating.

    Returns:
        * The stringified asm form.
        * An "identifier friendly" form of the shape and element type.
        * The raw element type.
    """
    rtt = RankedTensorType(tensor_type)
    ident = f"{'x'.join([str(dim) if dim >= 0 else 'D' for dim in rtt.shape])}x{rtt.element_type}"
    return str(rtt), ident, rtt.element_type


def specialize_all_known_dims(tensor_arg: TensorArg):
    """Specializes all dimensions of a tensor arg that are known.

    If a dimension is an `int`, it is specialized. Otherwise (i.e. SymInt) it
    is left dynamic.
    """
    spec_dims = tensor_arg.spec_dims
    for index, dim in enumerate(tensor_arg.t.shape):
        if isinstance(dim, int):
            spec_dims[index] = dim


@CustomOp.register()
class insert_slice(CustomOp):
    """
    Generates better IR for pytorch code like:
    `dst[indices] = src`
    where indices is a list of slices.

    Specify source tensor, destination tensor, offset, and strides for the insert_slice.
    """

    signature = (
        "insert_slice(Tensor src, Tensor dst, int[] offset, int[] strides) -> (Tensor)"
    )

    @no_type_check
    def select(self, ksel: KernelSelection):
        # tensor args
        src_desc = ksel.arg_tensor(0)
        specialize_all_known_dims(src_desc)
        dst_desc = ksel.arg_tensor(1)
        specialize_all_known_dims(dst_desc)
        # attr args
        offset_desc = ksel.attr_list_int(2)
        stride_desc = ksel.attr_list_int(3)

        torch._check(
            src_desc.t.dtype == dst_desc.t.dtype,
            lambda: f"Expected source and destination dtypes to match ({src_desc.t.dtype} != {dst_desc.t.dtype}).",
        )

        torch._check(
            len(offset_desc.v) == len(stride_desc.v) == len(dst_desc.t.shape),
            lambda: "Expected offset and stride lists to have length equal to the rank of destination tensor.",
        )

        result_desc = ksel.return_new_tensor(list(dst_desc.t.shape), dst_desc.t.dtype)
        specialize_all_known_dims(result_desc)

    @no_type_check
    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        dst_rtt = RankedTensorType(kb.arg_value(1).type)
        src_rtt = RankedTensorType(kb.arg_value(0).type)
        src_spec_dims = ksel.arg_descs[0].spec_dims
        offset = ksel.arg_descs[2].v
        stride = ksel.arg_descs[3].v
        sizes = []
        dynamic_dim_lines = ""
        for dim, spec_val in enumerate(src_spec_dims):
            if spec_val:
                sizes.append(spec_val)
                continue
            dim_value_name = f"%src_dim_{dim}"
            sizes.append(dim_value_name)
            dynamic_dim_lines += f"%cst_{dim} = arith.constant {dim} : index\n"
            dynamic_dim_lines += (
                f"{dim_value_name} = tensor.dim %src, %cst_{dim} : {src_rtt}\n"
            )

        identifier = (
            lambda rtt: f"{'x'.join([str(dim) if dim >= 0 else 'D' for dim in rtt.shape])}x{rtt.element_type}"
        )
        attr_id = lambda attr_list: "_".join([str(item) for item in attr_list])
        spec_sig = f"{identifier(src_rtt)}_into_{identifier(dst_rtt)}_{attr_id(offset)}_offset_{attr_id(stride)}_stride"
        function_name = f"insert_slice_{spec_sig}"

        func_op = _templates.inline_template_function(
            kb,
            "insert_slice",
            function_name,
            spec_sig=spec_sig,
            src_type=str(src_rtt),
            dst_type=str(dst_rtt),
            dynamic_dim_lines=dynamic_dim_lines,
            offset=str(offset),
            sizes=str(sizes),
            stride=str(stride),
        )

        arg_bindings = kb.arg_bindings[0:2]
        kb.yield_results(*impl_helper.call_function(func_op, *arg_bindings))
