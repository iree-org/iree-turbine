# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from typing import Sequence, Tuple

import torch

from ..conv_exports import ConvSignature, get_launchable, DEFAULT_LAYOUTS

__all__ = [
    "boo_conv",
    "enable_backward",
    "disable_backward",
]

BOO_USE_BACKWARD_KERNELS = int(os.getenv("BOO_USE_BACKWARD_KERNELS", "0"))


def enable_backward():
    """Allows toggling on Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 1


def disable_backward():
    """Allows toggling off Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 0


@torch.library.custom_op("iree_turbine::boo_convolution", mutates_args=())
def boo_convolution(
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
    x = x.detach()
    w = w.detach()

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
    args = (x, w) if b is None else (x, w, b.detach())
    return conv(*args)


@boo_convolution.register_fake
def _(
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


@torch.library.custom_op(
    "iree_turbine::boo_convolution_backward",
    mutates_args=(),
    schema="(Tensor x, Tensor w, Tensor grad_output, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout, bool[] mask) -> (Tensor?, Tensor?, Tensor?)",
)
def _boo_convolution_backward(
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
        input_grad = bwd_conv(grad_output, w)

    if mask[1]:
        wrw_conv = get_launchable(ConvSignature.get(x, w, mode="wrw", **kwargs))
        weight_grad = wrw_conv(grad_output, x)

    if mask[2]:
        # TODO: use iree to perform the reduce sum?
        output_layout = output_layout
        reduce_dims = []
        for i, char in enumerate(output_layout):
            if char != "C":
                reduce_dims.append(i)
        bias_grad = torch.sum(grad_output, reduce_dims)

    return input_grad, weight_grad, bias_grad


@_boo_convolution_backward.register_fake
def _b(
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


def pytorch_convolution_backward(ctx, grad_output):
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


def boo_convolution_backward(ctx, grad_output):
    if not BOO_USE_BACKWARD_KERNELS:
        return pytorch_convolution_backward(ctx, grad_output)

    x, w = ctx.saved_tensors

    mask = tuple((ctx.needs_input_grad[i] for i in range(3)))

    input_grad, weight_grad, bias_grad = _boo_convolution_backward(
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
    return input_grad, weight_grad, bias_grad, None, None, None, None, None, None, None


def boo_convolution_context(
    ctx,
    inputs,
    output,
):
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
    ) = inputs

    ctx.save_for_backward(x.detach(), w.detach())
    ctx.stride = stride
    ctx.padding = padding
    ctx.dilation = dilation
    ctx.groups = groups
    ctx.input_layout = input_layout
    ctx.kernel_layout = kernel_layout
    ctx.output_layout = output_layout

    ctx.use_bias = b is not None


boo_convolution.register_autograd(
    boo_convolution_backward, setup_context=boo_convolution_context
)


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

    def listify(value) -> Sequence[int]:
        if isinstance(value, Sequence):
            return value
        if isinstance(value, int):
            return [value] * num_spatial_dims
        return list(value)

    _infer = lambda layout: shared_layout or layout or DEFAULT_LAYOUTS[num_spatial_dims]

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
        listify(stride),
        listify(padding),
        listify(dilation),
        groups,
        _infer(input_layout),
        _infer(kernel_layout),
        _infer(output_layout),
    )
