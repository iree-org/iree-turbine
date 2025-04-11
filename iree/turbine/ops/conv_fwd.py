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


@CustomOp.register()
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

        # Currently, the conv op and accumulator type is hardcoded.
        # It may be advantageous to eventually allow passing these as string attributes.
        kwargs["conv_op"] = "linalg.conv_2d_nhwc_fhwc"
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
        arg_bindings = kb.arg_bindings[0:2]
        kb.yield_results(*impl_helper.call_function(func_op, *arg_bindings))


@CustomOp.register()
class generic_conv(CustomOp):
    """
    A basic layout-customizable convolution op.

    The argument signature is:
        input tensor,
        weight tensor,
        stride int[],
        dilation int[],
        input_layout str,
        weight_layout str,
        output_layout str,
        output_shape list[]?

    Special Layout Characters:
        the character 'c' is always assumed to correspond to a reduction dimension in the input and weight
        the character 'g' is always assumed to be parallel and shared among input, weight, and output
        the character 'n' is always assumed to be parallel and shared among input and output
        the character 'f' is always assumed to be parallel and shared among weight and output

    Other Layout Characters:
        are considered 'spatial' and must be shared between all three layouts
        induce reduction iteration into the weight tensor
        induce parallel iteration into the output tensor
        induce `stride * output_iterator + dilation * weight_iterator` iteration into the input tensor

    Assumptions:
        layout lengths match tensor shapes
        static dims
        no bias
        no pad
        non-quantized
        floating point input and weight dtypes (bitwidth <= 32)
        float32 accumulator
    """

    signature = "generic_conv(Tensor x, Tensor w, int[] s, int[] d, str xl, str wl, str ol, int[]? os) -> (Tensor)"

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
        # layout attrs
        xl_desc = ksel.attr_str(4)
        wl_desc = ksel.attr_str(5)
        ol_desc = ksel.attr_str(6)

        xl = xl_desc.v
        xs = list(x_desc.t.shape)
        wl = wl_desc.v
        ws = list(w_desc.t.shape)
        ol = ol_desc.v

        try:
            os_desc = ksel.attr_list_int(7)
            os = os_desc.v
            assert len(os) == len(
                ol
            ), f"output size must match output layout. Got {os} for layout {ol}."
        except Exception as e:
            logger.debug(
                "Invalid output shape provided for generic_conv. Falling back to default output shape calculation. Failed with exception %s.",
                str(e),
            )
            os = [-1] * len(ol)

        torch._check(
            len(xl) == len(xs),
            lambda: f'Length of input layout should match rank. Got layout "{xl}" for tensor with shape {xs}.',
        )
        torch._check(
            len(wl) == len(ws),
            lambda: f'Length of weight layout should match rank. Got layout "{wl}" for tensor with shape {ws}.',
        )
        special_characters = {"n", "c", "g", "f"}
        x_spatial = set(xl).difference(special_characters)
        w_spatial = set(wl).difference(special_characters)
        o_spatial = set(ol).difference(special_characters)

        torch._check(
            x_spatial == w_spatial and x_spatial == o_spatial,
            lambda: f"Layout specifications do not have consistent spatial dim characters. Got {xl} for input, {wl} for weight, and {ol} for output.",
        )

        num_spatial_dims = len(x_spatial)

        torch._check(
            len(s_desc.v) == num_spatial_dims,
            lambda: f"Number of strides should match number of spatial dims in input. Got stride = {s_desc.v} for {num_spatial_dims} spatial dims.",
        )

        torch._check(
            len(d_desc.v) == num_spatial_dims,
            lambda: f"Number of dilations should match number of spatial dims in input. Got dilations = {d_desc.v} for {num_spatial_dims} spatial dims.",
        )

        s_map = {}
        s_index = 0
        d_map = {}
        d_index = 0
        for c in xl:
            if c in special_characters:
                continue
            s_map[c] = s_desc.v[s_index]
            s_index += 1
            d_map[c] = d_desc.v[d_index]
            d_index += 1

        xs_map = {c: d for c, d in zip(xl, xs)}
        ws_map = {c: d for c, d in zip(wl, ws)}
        sp_map = {}
        for c in special_characters:
            x_d = xs_map.get(c, None)
            w_d = ws_map.get(c, None)
            if x_d and w_d:
                torch._check(
                    x_d == w_d,
                    lambda: f"mismatched special character dim size for character {c} in input (shape={xs}, layout={xl}) and weight (shape={ws}, layout={wl}).",
                )
            sp_map[c] = x_d if x_d is not None else w_d

        result_shape = []
        for i, c in enumerate(ol):
            provided = os[i]
            if c in special_characters:
                size = sp_map.get(c, 1)
                torch._check(
                    size == provided or provided == -1,
                    lambda: f"Cannot modify output size at dim '{c}' ({i}). Got provided size {provided}, actual = {size}.",
                )
            else:
                size = (
                    provided
                    if provided != -1
                    else (((xs_map[c] - 1) - d_map[c] * (ws_map[c] - 1)) // s_map[c])
                    + 1
                )
            result_shape.append(size)

        torch._check(
            sum([s <= 0 for s in result_shape]) == 0,
            lambda: f"Expected all output dim sizes to be positive. Got {result_shape=}.",
        )

        result_desc = ksel.return_new_tensor(result_shape, torch.float32)
        result_desc.specialize_all_dims()

    @no_type_check
    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        # get the attr values as a dict
        attr_dict = {}
        attr_dict["S"] = ksel.arg_descs[2].v
        attr_dict["D"] = ksel.arg_descs[3].v
        xl = attr_dict["xl"] = ksel.arg_descs[4].v
        wl = attr_dict["wl"] = ksel.arg_descs[5].v
        ol = attr_dict["ol"] = ksel.arg_descs[6].v

        res_desc = ksel.result_descs[0]
        special_characters = {"n", "c", "g", "f"}

        # make an attr string for building the spec_sig
        attr_str = ""
        for name, l in attr_dict.items():
            sep_char = "x" if name in ["S", "D"] else ""
            repr = name if name in ["S", "D"] else ""
            attr_str += sep_char.join([str(v) for v in l])
            attr_str += f"{repr}_"

        # get the input types and identifiers
        asm_types = {}
        ids = {}
        x_ir_type = kb.arg_value(0).type
        w_ir_type = kb.arg_value(1).type
        asm_types["X"], ids["X"], _ = unpack_tensor_type(x_ir_type)
        asm_types["W"], ids["W"], _ = unpack_tensor_type(w_ir_type)
        id_str = ""
        for name, id in ids.items():
            id_str += f"{id}_"
        # construct the spec_sig and func name
        spec_sig = f"{id_str}{attr_str}f32"
        function_name = f"generic_conv_" + spec_sig
        # build the template inliner kwargs
        kwargs = {"spec_sig": spec_sig}
        # input arg types
        for name, ty in asm_types.items():
            kwargs[f"{name}_asm_type"] = ty

        # Currently, the accumulator type is hardcoded.
        # It may be advantageous to eventually allow passing this as a string attribute.
        kwargs["accum_dtype"] = "f32"
        kwargs[
            "result_asm_type"
        ] = f"tensor<{'x'.join('?' if d is None else str(d) for d in res_desc.spec_dims)}x{kwargs['accum_dtype']}>"

        kwargs["X_dtype"] = str(RankedTensorType(x_ir_type).element_type)
        kwargs["W_dtype"] = str(RankedTensorType(w_ir_type).element_type)

        # linalg.generic properties:
        iterator_order = list([char for char in ol])  # all output dims are parallel
        iterator_types = ["parallel"] * len(ol)
        for char in wl:
            if char in special_characters.intersection(iterator_order):
                continue
            iterator_type = (
                "parallel"
                if char in special_characters.difference({"c"})
                else "reduction"
            )
            uniqued_char = char if char == "c" else f"weight_{char}"
            iterator_order.append(uniqued_char)
            iterator_types.append(iterator_type)
        for char in xl:
            if char not in special_characters or char in iterator_order:
                continue
            iterator_type = "reduction" if char == "c" else "parallel"
            iterator_order.append(char)
            iterator_types.append(iterator_type)
        d_mapping = {char: f"d{i}" for i, char in enumerate(iterator_order)}
        num_ds = len(iterator_order)
        base_mapping_str = "(" + ", ".join(list([f"d{i}" for i in range(num_ds)])) + ")"

        x_indexing = []
        spatial_dim_idx = 0
        for char in xl:
            if char in special_characters:
                x_indexing.append(d_mapping[char])
                continue
            weight_spatial = d_mapping[f"weight_{char}"]
            stride = attr_dict["S"][spatial_dim_idx]
            dilation = attr_dict["D"][spatial_dim_idx]
            lhs = d_mapping[char] if stride == 1 else f"{stride}*{d_mapping[char]}"
            rhs = weight_spatial if dilation == 1 else f"{dilation}*{weight_spatial}"
            x_indexing.append(f"{lhs} + {rhs}")
            spatial_dim_idx += 1

        w_indexing = []
        for char in wl:
            if char in special_characters:
                w_indexing.append(d_mapping[char])
                continue
            w_indexing.append(d_mapping[f"weight_{char}"])
        result_indexing = list([f"d{i}" for i in range(len(ol))])

        def to_indexing_map(indexing):
            map_string = f"affine_map<{base_mapping_str} -> ("
            map_string += ", ".join(indexing)
            map_string += ")>"
            return map_string

        kwargs["iterator_types"] = str(iterator_types).replace("'", '"')
        kwargs["X_indexing_map"] = to_indexing_map(x_indexing)
        kwargs["W_indexing_map"] = to_indexing_map(w_indexing)
        kwargs["result_indexing_map"] = to_indexing_map(result_indexing)

        func_op = _templates.inline_template_function(
            kb,
            "generic_conv",
            function_name,
            **kwargs,
        )
        arg_bindings = kb.arg_bindings[0:2]
        kb.yield_results(*impl_helper.call_function(func_op, *arg_bindings))
