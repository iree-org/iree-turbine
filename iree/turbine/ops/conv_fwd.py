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
    def_library,
    impl_helper,
)

LIBRARY = def_library("turbine_ops")
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


@CustomOp.register(library=LIBRARY)
class conv_2d_nhwc_fhwc(CustomOp):
    """
    Basic 2-D channels-last forward convolution op.

    Input format  : N x H x W x C.
    Weight format : F x Hk x Wk x C.
    Return format : N x Ho x Wo x F, dtype=f32.

    Assumptions:
        static dims
        no bias
        no pad
        non-quantized
        equal input and weight floating point dtypes
        float32 accumulator

    The argument signature is: input tensor, weight tensor, list of stride ints, list of dilation ints.
    """

    signature = "conv_2d_nhwc_fhwc(Tensor x, Tensor w, int[] s, int[] d) -> (Tensor)"

    @no_type_check
    def select(self, ksel: KernelSelection):
        # tensor args
        x_desc = ksel.arg_tensor(0)
        x_desc.specialize_all_dims()
        w_desc = ksel.arg_tensor(1)
        w_desc.specialize_all_dims()
        # attr args
        s_desc = ksel.attr_list_int(2)
        d_desc = ksel.attr_list_int(3)

        # assume 2d and NHWC for now
        n, *spatial, c_x = x_desc.t.shape
        f, *spatial_k, c_k = w_desc.t.shape

        # TODO: check that the input shapes and dtypes are valid
        # TODO: check that the number of strides and dilations are valid

        spatial_o = []
        for i, k, s, d in zip(spatial, spatial_k, s_desc.v, d_desc.v):
            spatial_o.append((((i - 1) - d * (k - 1)) // s) + 1)

        # Build the result description
        result_shape = [n]
        result_shape.extend(spatial_o)
        result_shape.append(f)
        result_desc = ksel.return_new_tensor(result_shape, torch.float32)
        result_desc.specialize_all_dims()

    @no_type_check
    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        res_desc = ksel.result_descs[0]
        spatial_dims = len(res_desc.t.shape) - 2
        # get the attr values as a dict
        attr_dict = {}
        attr_dict["S"] = ksel.arg_descs[2].v
        attr_dict["D"] = ksel.arg_descs[3].v
        # make an attr string for building the spec_sig
        attr_str = ""
        for name, l in attr_dict.items():
            attr_str += "x".join([str(v) for v in l])
            attr_str += f"{name}_"
        # get the input types and identifiers
        asm_types = {}
        ids = {}
        asm_types["X"], ids["X"], _ = unpack_tensor_type(kb.arg_value(0).type)
        asm_types["W"], ids["W"], _ = unpack_tensor_type(kb.arg_value(1).type)
        id_str = ""
        for name, id in ids.items():
            id_str += f"{name}{id}_"
        # construct the spec_sig and func name
        spec_sig = f"{id_str}{attr_str}f32"
        function_name = "conv_2d_nhwc_fhwc_" + spec_sig
        # build the template inliner kwargs
        kwargs = {"spec_sig": spec_sig}
        # input arg types
        for name, ty in asm_types.items():
            kwargs[f"{name}_asm_type"] = ty
        # attr values
        for name, l in attr_dict.items():
            for i, v in enumerate(l):
                kwargs[f"{name}{i}"] = v
        # output dim values
        for i in range(spatial_dims):
            v = res_desc.t.shape[i + 1]
            kwargs[f"OUT_DIM{i}"] = v

        kwargs["conv_op"] = f"linalg.conv_2d_nhwc_fhwc"
        kwargs["accum_dtype"] = "f32"
        kwargs[
            "result_asm_type"
        ] = f"tensor<{'x'.join('?' if d is None else str(d) for d in res_desc.spec_dims)}x{kwargs['accum_dtype']}>"

        func_op = _templates.inline_template_function(
            kb,
            "conv_2d_nhwc_fhwc",
            function_name,
            **kwargs,
        )
        print(func_op)
        arg_bindings = kb.arg_bindings[0:2]
        kb.yield_results(*impl_helper.call_function(func_op, *arg_bindings))
