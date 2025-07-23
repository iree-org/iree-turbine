# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence, Tuple

import torch

from .library import register_meta, BOO_LIBRARY
from .utils import *

from ..op_exports.conv import (
    ConvSignature,
    DEFAULT_LAYOUTS,
    get_conv_func_name,
    Permutation,
)

from ..driver.launch import get_launchable
from ..runtime import LaunchableRuntimeCache

from ....runtime.op_reg import CustomOp, KernelBuilder, KernelSelection, impl_helper
from ....transforms.merger import Merger
from ....support.ir_imports import Operation, StringAttr

__all__ = [
    "boo_layout_customizable_convolution",
    "convolution_replacement",
]

# Forward Convolution Implementations #


@CustomOp.register(library=BOO_LIBRARY)
class layout_customizable_convolution(CustomOp):
    signature = "layout_customizable_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout) -> Tensor"

    def select(self, ksel: KernelSelection):
        # Declare args.
        input = ksel.arg_tensor(0)
        weight = ksel.arg_tensor(1)
        bias = ksel.arg_optional_tensor(2)
        stride = ksel.attr_list_int(3)
        padding = ksel.attr_list_int(4)
        dilation = ksel.attr_list_int(5)
        groups = ksel.attr_int(6)
        input_layout = ksel.attr_str(7)
        kernel_layout = ksel.attr_str(8)
        output_layout = ksel.attr_str(9)
        # Specialize args.
        input.specialize_all_dims()
        weight.specialize_all_dims()
        if bias:
            bias.specialize_all_dims()
            ksel.variant = "biased"

        self.conv_sig = ConvSignature(
            input_shape=input.spec_dims,
            kernel_shape=weight.spec_dims,
            bias=(bias is not None),
            input_layout=input_layout.v,
            kernel_layout=kernel_layout.v,
            output_layout=output_layout.v,
            dtype=input.t.dtype,
            stride=stride.v,
            padding=padding.v,
            dilation=dilation.v,
            groups=groups.v,
        )
        output_shape = self.conv_sig.output_shape
        o_meta = torch.empty(
            tuple(output_shape), dtype=self.conv_sig.dtype, device="meta"
        )
        ksel.return_tensor(o_meta)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        sample_args = (ksel.arg_descs[0].t, ksel.arg_descs[1].t)
        if ksel.variant == "biased":
            sample_args = sample_args + (ksel.arg_descs[2].t,)
        func_name = self.conv_sig.func_name
        # Get a module containing the func op for our custom convolution.
        # This IR is a combination of expanded CustomOps and torch code.
        # We are essentially fusing these things together into one inline-able op.
        module_op = generate_custom_op_compatible_ir(
            self.conv_sig.get_nn_module(use_custom=True),
            args=sample_args,
            func_name=func_name,
            context=kb.context,
        )
        merger = Merger(
            module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
        )
        merger.merge()
        func_op = kb.symbol_table[merger.translate_symbol(func_name)]
        kb.yield_results(
            *impl_helper.call_function(
                func_op,
                *[binding for binding in kb.arg_bindings if binding is not None]
            )
        )

    def eager_execute(self, *args):
        return _boo_layout_customizable_convolution_impl(*args)


def _boo_layout_customizable_convolution_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    b: None | torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
) -> torch.Tensor:

    # Unfortunately, pytorch converts the tuple inputs to lists for some reason.
    # We need to convert them back to tuples.
    func_name = get_conv_func_name(
        tuple(x.shape),
        tuple(w.shape),
        str(x.dtype),
        "FORWARD",
        (b is not None),
        tuple(stride),
        tuple(padding),
        tuple(dilation),
        groups,
        input_layout,
        kernel_layout,
        output_layout,
    )
    args = (x.data, w.data) if b is None else (x.data, w.data, b.data)
    cache_hit = LaunchableRuntimeCache.get(func_name)
    if cache_hit:
        return cache_hit(*args)

    sig = ConvSignature(
        input_shape=x.shape,
        kernel_shape=w.shape,
        input_layout=input_layout,
        kernel_layout=kernel_layout,
        output_layout=output_layout,
        bias=(b is not None),
        dtype=x.dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=False,
        output_padding=0,
        groups=groups,
        mode="fwd",
    )

    # Get a launchable and apply.
    conv = get_launchable(sig)
    return conv(*args)


# Backward Convolution Implementations #
@CustomOp.register(library=BOO_LIBRARY, register_meta=False)
class layout_customizable_convolution_backward(CustomOp):
    signature = "layout_customizable_convolution_backward(Tensor input, Tensor weight, Tensor grad_output, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout, bool[] mask) -> (Tensor?, Tensor?, Tensor?)"

    def select(self, ksel):
        raise NotImplementedError("convolution_backward select NYI")

    def generate(self, ksel, kb):
        raise NotImplementedError("convolution_backward generate NYI")

    def eager_execute(self, *args):
        return _boo_layout_customizable_convolution_backward_impl(*args)


def _boo_layout_customizable_convolution_backward_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_output: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
    mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "input_layout": input_layout,
        "kernel_layout": kernel_layout,
        "output_layout": output_layout,
    }

    input_grad = weight_grad = bias_grad = None

    if mask[0]:
        bwd_sig = ConvSignature.get(x, w, mode="bwd", **kwargs)
        bwd_conv = get_launchable(bwd_sig)
        input_grad = bwd_conv(grad_output, w.data)

    if mask[1]:
        wrw_conv = get_launchable(ConvSignature.get(x, w, mode="wrw", **kwargs))
        weight_grad = wrw_conv(grad_output, x.data)

    if mask[2]:
        # TODO: use iree to perform the reduce sum?
        output_layout = output_layout
        reduce_dims = []
        for i, char in enumerate(output_layout):
            if char != "C":
                reduce_dims.append(i)
        bias_grad = torch.sum(grad_output, reduce_dims)

    return input_grad, weight_grad, bias_grad


@register_meta("layout_customizable_convolution_backward")
def _boo_layout_customizable_convolution_backward_meta(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_output: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
    mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    input_grad = weight_grad = bias_grad = None
    if mask[0]:
        input_grad = torch.empty_like(x)
    if mask[1]:
        weight_grad = torch.empty_like(w)
    if mask[2]:
        output_channels = w.shape[kernel_layout.find("N")]
        bias_grad = torch.empty([output_channels], dtype=x.dtype, device=x.device)
    return input_grad, weight_grad, bias_grad


def pytorch_layout_customizable_convolution_backward(ctx, grad_output):
    """Fallback implementation for backward."""
    x, w = ctx.saved_tensors

    mask = tuple((ctx.needs_input_grad[i] for i in range(3)))

    # return to NCHW if necessary
    num_spatial_dims = len(x.shape) - 2
    default_layout = DEFAULT_LAYOUTS[num_spatial_dims]
    input_perm = None
    kernel_perm = None
    output_perm = None
    if ctx.input_layout != default_layout:
        input_perm = Permutation.get(ctx.input_layout, default_layout)
        x = input_perm(x)
    if ctx.kernel_layout != default_layout:
        kernel_perm = Permutation.get(ctx.kernel_layout, default_layout)
        w = kernel_perm(w)
    if ctx.output_layout != default_layout:
        output_perm = Permutation.get(ctx.output_layout, default_layout)
        grad_output = output_perm(grad_output)

    input_grad, weight_grad, bias_grad = torch.ops.aten.convolution_backward(
        grad_output,
        x,
        w,
        [w.shape[0]],
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        False,
        [0] * len(ctx.stride),
        ctx.groups,
        mask,
    )

    if input_perm is not None and mask[0]:
        input_grad = input_perm.inv()(input_grad)
    if kernel_perm is not None and mask[1]:
        weight_grad = kernel_perm.inv()(weight_grad)
    # return `None` for attribute args
    return input_grad, weight_grad, bias_grad, None, None, None, None, None, None, None


# Autograd Implementation #


class _Boolayout_customizable_Convolution(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        (
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
            input_layout,
            kernel_layout,
            output_layout,
        ) = args

        ctx.save_for_backward(x, w)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_layout = input_layout
        ctx.kernel_layout = kernel_layout
        ctx.output_layout = output_layout

        ctx.use_bias = b is not None

        return torch.ops.boo.layout_customizable_convolution(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
            input_layout,
            kernel_layout,
            output_layout,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if not is_boo_backward_enabled():
            return pytorch_layout_customizable_convolution_backward(ctx, grad_output)

        x, w = ctx.saved_tensors

        mask = tuple((ctx.needs_input_grad[i] for i in range(3)))

        (
            input_grad,
            weight_grad,
            bias_grad,
        ) = torch.ops.boo.layout_customizable_convolution_backward(
            x,
            w,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
            ctx.input_layout,
            ctx.kernel_layout,
            ctx.output_layout,
            mask,
        )

        # return `None` for attribute args
        return (
            input_grad,
            weight_grad,
            bias_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def boo_layout_customizable_convolution(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
) -> torch.Tensor:
    """A fully layout-customizable convolution. Inputs x, and w are expected to have appropriate shapes for the provided layouts."""
    use_autograd = torch._C.is_grad_enabled() and (
        w.requires_grad or x.requires_grad or (b is not None and b.requires_grad)
    )
    return (
        _Boolayout_customizable_Convolution.apply(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
            input_layout,
            kernel_layout,
            output_layout,
        )
        if use_autograd
        else torch.ops.boo.layout_customizable_convolution(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
            input_layout,
            kernel_layout,
            output_layout,
        )
    )


def convolution_replacement(
    x: torch.Tensor,
    w: torch.Tensor,
    b: None | torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    output_is_channels_last: bool | None = None,
) -> torch.Tensor:
    """Intended to be used for replacing `torch.ops.aten.convolution` in an fx.Graph to generate better IR for fusions.

    For eager boo convolution, use boo_conv from iree.turbine.kernel.boo.ops.conv instead.
    """
    num_spatial_dims = len(x.shape) - 2

    mem_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_spatial_dims)
    default_layout = DEFAULT_LAYOUTS[num_spatial_dims]
    cl_layout = CHANNELS_LAST_LAYOUTS[num_spatial_dims]
    cl_contig_perm = CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION[num_spatial_dims]
    contig_cl_perm = CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION[num_spatial_dims]

    x_cl = False if mem_format is None else x.is_contiguous(memory_format=mem_format)
    w_cl = False if mem_format is None else w.is_contiguous(memory_format=mem_format)

    input_layout = cl_layout if x_cl else default_layout
    kernel_layout = cl_layout if w_cl else default_layout
    output_is_channels_last = (
        output_is_channels_last if output_is_channels_last is not None else w_cl
    )
    output_layout = cl_layout if output_is_channels_last else default_layout

    x = x if not x_cl else x.permute(cl_contig_perm)
    w = w if not w_cl else w.permute(cl_contig_perm)

    result: torch.Tensor = layout_customizable_convolution(
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        groups,
        input_layout=input_layout,
        kernel_layout=kernel_layout,
        output_layout=output_layout,
    )
    return result if not output_is_channels_last else result.permute(contig_cl_perm)
