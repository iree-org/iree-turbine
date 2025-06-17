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

__all__ = [
    "boo_layout_customizable_convolution",
]

# Forward Convolution Implementations #

define_schema(
    "layout_customizable_convolution",
    "(Tensor x, Tensor w, Tensor? b, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout) -> Tensor",
)


@register_impl("layout_customizable_convolution")
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


@register_meta("layout_customizable_convolution")
def _boo_layout_customizable_convolution_meta(
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
    return torch.empty(sig.output_shape, dtype=sig.dtype, device=x.device)


# Backward Convolution Implementations #

define_schema(
    "layout_customizable_convolution_backward",
    "(Tensor x, Tensor w, Tensor grad_output, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout, bool[] mask) -> (Tensor?, Tensor?, Tensor?)",
)


@register_impl("layout_customizable_convolution_backward")
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
    rank = len(x.shape)
    perm = [0] + [rank - 1] + list(range(1, rank - 1))
    inv_perm = [0] + list(range(2, rank)) + [1]
    if ctx.input_layout.endswith("C"):
        x = x.permute(perm)
    if ctx.kernel_layout.endswith("C"):
        w = w.permute(perm)
    if ctx.output_layout.endswith("C"):
        grad_output = grad_output.permute(perm)

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

    if ctx.input_layout.endswith("C") and mask[0]:
        input_grad = input_grad.permute(inv_perm)
    if ctx.kernel_layout.endswith("C") and mask[1]:
        weight_grad = weight_grad.permute(inv_perm)
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
    b: torch.Tensor,
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
