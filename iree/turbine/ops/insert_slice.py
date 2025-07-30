# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Tuple, no_type_check, overload
import torch

from ..support.ir_imports import (
    Value,
    IndexType,
    RankedTensorType,
    IrType,
    linalg_d,
    FloatType,
    tensor_d,
    arith_d,
    IntegerAttr,
    IntegerType,
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

    @property
    def signature(self):
        return "insert_slice(Tensor src, Tensor dst, int[] offset, int[] strides) -> (Tensor)"

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


@CustomOp.register()
class generic_insert_slice(CustomOp):
    """
    Generates a `linalg.generic` for pytorch code like:
    `dst[indices] = src`
    where indices is a list of slices.

    Specify source tensor, sizes, offsets, and strides for the insert_slice.
    """

    @property
    def signature(self):
        return "generic_insert_slice(Tensor src, int[] sizes, int[] offset, int[] strides) -> (Tensor)"

    @no_type_check
    def select(self, ksel: KernelSelection):
        # tensor args
        src_desc = ksel.arg_tensor(0)
        src_desc.specialize_all_dims()
        # attr args
        sizes_desc = ksel.attr_list_int(1)
        offsets_desc = ksel.attr_list_int(2)
        strides_desc = ksel.attr_list_int(3)

        has_offsets = any([o != 0 for o in offsets_desc.v])
        indexing_valid = all(
            [
                o + in_size * s in range(out_size, out_size + s)
                for o, in_size, s, out_size in zip(
                    offsets_desc.v, src_desc.t.shape, strides_desc.v, sizes_desc.v
                )
            ]
        )

        ksel.variant = (
            "tensor_extract" if has_offsets or not indexing_valid else "default"
        )

        torch._check(
            len(offsets_desc.v) == len(strides_desc.v) == len(sizes_desc.v),
            lambda: "Expected offset and stride lists to have length equal to the rank of destination tensor.",
        )

        result_desc = ksel.return_new_tensor(sizes_desc.v, src_desc.t.dtype)
        result_desc.specialize_all_dims()

    @no_type_check
    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        src_rtt = RankedTensorType(kb.arg_value(0).type)
        sizes = ksel.arg_descs[1].v
        offset = ksel.arg_descs[2].v
        stride = ksel.arg_descs[3].v
        arg_binding = kb.arg_bindings[0]
        element_type = RankedTensorType(arg_binding.type).element_type

        def body_builder_direct(arg: element_type, out: element_type) -> element_type:  # type: ignore
            indices = [
                linalg_d.IndexOp(
                    IntegerAttr.get(IntegerType.get_signless(64), i)
                ).result
                for i in range(len(sizes))
            ]
            checks = []
            for i, s in zip(indices, stride):
                if s == 1:
                    continue
                rem = arith_d.RemUIOp(i, arith_d.constant(IndexType.get(), s))
                cmp = arith_d.CmpIOp(
                    arith_d.CmpIPredicate.eq, rem, arith_d.constant(IndexType.get(), 0)
                )
                checks.append(cmp)
            zero = arith_d.constant(
                element_type, 0.0 if isinstance(element_type, FloatType) else 0
            )
            output = arg
            for check in checks:
                output = arith_d.select(check, output, zero)
            return output

        # I'm not sure if this is going to codegen well. Should test and look at alternatives.
        def body_builder_tensor_extract(out: element_type) -> element_type:  # type: ignore
            indices = [
                linalg_d.IndexOp(
                    IntegerAttr.get(IntegerType.get_signless(64), i)
                ).result
                for i in range(len(sizes))
            ]
            checks = []
            extract_indices = []
            zero_index = arith_d.constant(IndexType.get(), 0)
            for i, s, o, input_dim in zip(indices, stride, offset, src_rtt.shape):
                stride_index = arith_d.constant(IndexType.get(), s)
                offset_index = arith_d.constant(IndexType.get(), o)
                # i - offset
                sub = arith_d.subi(i, offset_index)
                # (i - offset) % stride
                rem = arith_d.RemUIOp(sub, stride_index)
                # (i - offset) % stride == 0
                cmp = arith_d.CmpIOp(arith_d.CmpIPredicate.eq, rem, zero_index)
                checks.append(cmp)
                # (i - offset) // stride
                input_index = arith_d.FloorDivSIOp(sub, stride_index)
                # clamp input index to [0, input_size - 1]
                input_index_geq_0 = arith_d.CmpIOp(
                    arith_d.CmpIPredicate.sge, input_index, zero_index
                )
                cst_input_dim_m_1 = arith_d.constant(IndexType.get(), input_dim - 1)
                input_index_leq_input_dim_m_1 = arith_d.CmpIOp(
                    arith_d.CmpIPredicate.sle, input_index, cst_input_dim_m_1
                )
                clamp0 = arith_d.select(input_index_geq_0, input_index, zero_index)
                clamp1 = arith_d.select(
                    input_index_leq_input_dim_m_1, clamp0, cst_input_dim_m_1
                )
                extract_indices.append(clamp1)
                # Return zero when outside clamp
                checks.extend([input_index_geq_0, input_index_leq_input_dim_m_1])

            zero = arith_d.constant(
                element_type, 0.0 if isinstance(element_type, FloatType) else 0
            )
            output = tensor_d.extract(arg_binding, extract_indices)
            for check in checks:
                output = arith_d.select(check, output, zero)
            return output

        with kb.ip and kb.context:
            empty = tensor_d.EmptyOp(
                sizes=sizes, element_type=src_rtt.element_type
            ).result
            in_exprs: list[linalg_d.AffineExpr] = []
            out_exprs: list[linalg_d.AffineExpr] = []
            for i in range(len(sizes)):
                dim_expr = linalg_d.AffineDimExpr.get(i)
                out_exprs.append(dim_expr)
                in_exprs.append(
                    dim_expr
                    if stride[i] == 1
                    else linalg_d.AffineExpr.get_floor_div(
                        dim_expr, linalg_d.AffineExpr.get_constant(stride[i])
                    )
                )
            in_map = linalg_d.AffineMap.get(
                dim_count=len(sizes), symbol_count=0, exprs=in_exprs
            )
            out_map = linalg_d.AffineMap.get(
                dim_count=len(sizes), symbol_count=0, exprs=out_exprs
            )
            inputs = (arg_binding,) if ksel.variant == "default" else ()
            outputs = (empty,)
            maps = (in_map, out_map) if ksel.variant == "default" else (out_map,)
            iterator_types = len(sizes) * ("parallel",)
            body_builder = (
                body_builder_direct
                if ksel.variant == "default"
                else body_builder_tensor_extract
            )
            g: Value = linalg_d.generic(inputs, outputs, maps, iterator_types)(
                body_builder
            )

        kb.yield_results(g)
