# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Sequence, Tuple

import torch

from ..conv_exports import (
    ConvSignature,
    get_launchable,
    DEFAULT_LAYOUTS,
    ConvLaunchableRuntimeCache,
)

from .library import define_schema, register_impl, register_meta

from .utils import *
from .layout_customizable_conv import boo_layout_customizable_convolution

__all__ = [
    "boo_conv",
    "boo_convolution",
]

# Forward Convolution Implementations #

define_schema(
    "convolution",
    "(Tensor x, Tensor w, Tensor? b, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor",
)


@register_impl("convolution")
def _boo_convolution_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    b: None | torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:

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
    # Match output layout to weight layout to propagate channels_last format.
    output_layout = cl_layout if w_cl else default_layout

    x = x if not x_cl else x.permute(cl_contig_perm)
    w = w if not w_cl else w.permute(cl_contig_perm)

    # Unfortunately, pytorch converts the tuple inputs to lists for some reason.
    # We need to convert them back to tuples.
    func_name = get_func_name(
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
    cache_hit = ConvLaunchableRuntimeCache.get(func_name)
    if cache_hit:
        result = cache_hit(*args)
        return result if not w_cl else result.permute(contig_cl_perm)

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
    result = conv(*args)
    return result if not w_cl else result.permute(contig_cl_perm)


@register_meta("convolution")
def _boo_convolution_meta(
    x: torch.Tensor,
    w: torch.Tensor,
    b: None | torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    sig = ConvSignature(
        input_shape=x.shape,
        kernel_shape=w.shape,
        dtype=x.dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=False,
        output_padding=0,
        groups=groups,
        mode="fwd",
    )
    num_spatial_dims = len(x.shape) - 2
    cl_memory_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_spatial_dims)
    memory_format = (
        cl_memory_format
        if cl_memory_format is not None
        and w.is_contiguous(memory_format=cl_memory_format)
        else torch.contiguous_format
    )
    return torch.empty(
        sig.output_shape, dtype=sig.dtype, device=x.device, memory_format=memory_format
    )


# Backward Convolution Implementations #

define_schema(
    "convolution_backward",
    "(Tensor x, Tensor w, Tensor grad_output, int[] stride, int[] padding, int[] dilation, int groups, bool[] mask) -> (Tensor?, Tensor?, Tensor?)",
)


@register_impl("convolution_backward")
def _boo_convolution_backward_impl(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_output: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

    num_spatial_dims = len(x.shape) - 2
    mem_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_spatial_dims)
    default_layout = DEFAULT_LAYOUTS[num_spatial_dims]
    cl_layout = CHANNELS_LAST_LAYOUTS[num_spatial_dims]
    cl_contig_perm = CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION[num_spatial_dims]
    contig_cl_perm = CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION[num_spatial_dims]

    x_cl = False if mem_format is None else x.is_contiguous(memory_format=mem_format)
    w_cl = False if mem_format is None else w.is_contiguous(memory_format=mem_format)
    o_cl = (
        False
        if mem_format is None
        else grad_output.is_contiguous(memory_format=mem_format)
    )

    input_layout = cl_layout if x_cl else default_layout
    kernel_layout = cl_layout if w_cl else default_layout
    output_layout = cl_layout if o_cl else default_layout

    x = x if not x_cl else x.permute(cl_contig_perm)
    w = w if not w_cl else w.permute(cl_contig_perm)
    grad_output = grad_output if not o_cl else grad_output.permute(cl_contig_perm)

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
        input_grad = input_grad if not x_cl else input_grad.permute(contig_cl_perm)

    if mask[1]:
        wrw_conv = get_launchable(ConvSignature.get(x, w, mode="wrw", **kwargs))
        weight_grad = wrw_conv(grad_output, x.data)
        weight_grad = weight_grad if not w_cl else weight_grad.permute(contig_cl_perm)

    if mask[2]:
        # TODO: use iree to perform the reduce sum?
        output_layout = output_layout
        reduce_dims = []
        for i, char in enumerate(output_layout):
            if char != "C":
                reduce_dims.append(i)
        bias_grad = torch.sum(grad_output, reduce_dims)

    return input_grad, weight_grad, bias_grad


@register_meta("convolution_backward")
def _boo_convolution_backward_meta(
    x: torch.Tensor,
    w: torch.Tensor,
    grad_output: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    input_grad = weight_grad = bias_grad = None
    if mask[0]:
        input_grad = torch.empty_like(x)
    if mask[1]:
        weight_grad = torch.empty_like(w)
    if mask[2]:
        output_channels = w.shape[0]
        bias_grad = torch.empty([output_channels], dtype=x.dtype, device=x.device)
    return input_grad, weight_grad, bias_grad


def pytorch_convolution_backward(ctx, grad_output):
    """Fallback implementation for backward."""
    x, w = ctx.saved_tensors

    mask = tuple((ctx.needs_input_grad[i] for i in range(3)))

    # return to NCHW if necessary
    input_grad, weight_grad, bias_grad = torch.ops.aten.convolution_backward(
        grad_output,
        x,
        w,
        None,
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        False,
        [0] * len(ctx.stride),
        ctx.groups,
        mask,
    )

    # return `None` for attribute args
    return input_grad, weight_grad, bias_grad, None, None, None, None


# Autograd Implementation #


class _BooConvolution(torch.autograd.Function):
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
        ) = args

        ctx.save_for_backward(x, w)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return torch.ops.boo.convolution(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
        )

    @staticmethod
    def backward(ctx, grad_output):
        if not is_boo_backward_enabled():
            return pytorch_convolution_backward(ctx, grad_output)

        x, w = ctx.saved_tensors

        mask = tuple((ctx.needs_input_grad[i] for i in range(3)))

        input_grad, weight_grad, bias_grad = torch.ops.boo.convolution_backward(
            x,
            w,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
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
        )


def boo_convolution(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
) -> torch.Tensor:
    """Similar to boo_conv, but does not pre-process, nor provide defaults for, arguments like stride, dilation, etc."""
    use_autograd = torch._C.is_grad_enabled() and (
        w.requires_grad or x.requires_grad or (b is not None and b.requires_grad)
    )
    return (
        _BooConvolution.apply(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
        )
        if use_autograd
        else torch.ops.boo.convolution(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            groups,
        )
    )


# Lazy Autograd Implementation #


def boo_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
):
    """
    Applies a differentiable forward convolution kernel.

    kwargs can include any of the following usual convolution options:

        stride         : int or int[]
        padding        : int or int[]
        dilation       : int or int[]
        groups         : int
    """

    num_spatial_dims = len(weight.shape) - 2

    # The decorators torch.amp.custom_fwd/custom_bwd don't seem to work with torch.library.custom_op
    # For now, this is a quick hack to manually do the casting outside our custom op.
    device_type = input.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        input = input.to(dtype=dtype)
        weight = weight.to(dtype=dtype)
        bias = bias if bias is None else bias.to(dtype=dtype)

    return boo_convolution(
        input,
        weight,
        bias,
        make_tuple(stride, num_spatial_dims),
        make_tuple(padding, num_spatial_dims),
        make_tuple(dilation, num_spatial_dims),
        groups,
    )


# Lazy Autograd Implementation #


def boo_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    shared_layout: str | None = None,
    input_layout: str | None = None,
    kernel_layout: str | None = None,
    output_layout: str | None = None,
):
    """
    Applies a differentiable forward convolution kernel.

    kwargs can include any of the following usual convolution options:

        stride         : int or int[]
        padding        : int or int[]
        dilation       : int or int[]
        groups         : int

    Users can also specify alternative layouts for each convolution:

        shared_layout  : str
        input_layout   : str
        kernel_layout  : str
        output_layout  : str

    These layouts should be permutations of "NCH", "NCHW", or "NCDHW".
    """

    num_spatial_dims = len(weight.shape) - 2
    no_layouts = all(
        [x is None for x in [shared_layout, input_layout, kernel_layout, output_layout]]
    )

    # The decorators torch.amp.custom_fwd/custom_bwd don't seem to work with torch.library.custom_op
    # For now, this is a quick hack to manually do the casting outside our custom op.
    device_type = input.device.type
    if torch.is_autocast_enabled(device_type):
        dtype = torch.get_autocast_dtype(device_type)
        input = input.to(dtype=dtype)
        weight = weight.to(dtype=dtype)
        bias = bias if bias is None else bias.to(dtype=dtype)

    if no_layouts:
        return boo_convolution(
            input,
            weight,
            bias,
            make_tuple(stride, num_spatial_dims),
            make_tuple(padding, num_spatial_dims),
            make_tuple(dilation, num_spatial_dims),
            groups,
        )

    _infer = lambda layout: shared_layout or layout or DEFAULT_LAYOUTS[num_spatial_dims]

    return boo_layout_customizable_convolution(
        input,
        weight,
        bias,
        make_tuple(stride, num_spatial_dims),
        make_tuple(padding, num_spatial_dims),
        make_tuple(dilation, num_spatial_dims),
        groups,
        _infer(input_layout),
        _infer(kernel_layout),
        _infer(output_layout),
    )
