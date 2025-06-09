# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import functools

from typing import Sequence, Tuple, Any, Iterable

import torch

from ..conv_exports import (
    ConvSignature,
    get_launchable,
    DEFAULT_LAYOUTS,
    ConvLaunchableRuntimeCache,
)

from .library import define_schema, register_impl, register_meta

__all__ = [
    "boo_conv",
    "boo_convolution",
    "make_tuple",
    "enable_backward",
    "disable_backward",
]

# Toggle Using Boo Backward Kernels #

BOO_USE_BACKWARD_KERNELS = int(os.getenv("BOO_USE_BACKWARD_KERNELS", "0"))


def enable_backward():
    """Allows toggling on Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 1


def disable_backward():
    """Allows toggling off Boo backward convolution kernels from python."""
    global BOO_USE_BACKWARD_KERNELS
    BOO_USE_BACKWARD_KERNELS = 0


# Utilities #


def make_tuple(a: Any, size: int) -> Tuple:
    """Tries to convert `a` into a Tuple of ints."""
    if isinstance(a, Iterable):
        result = tuple(a)
        assert len(result) == size
        assert isinstance(result[0], int)
        return result
    if isinstance(a, int):
        return (a,) * size
    raise TypeError(f"Input {a} is expected to be an iterable or int. Got {type(a)}.")


@functools.lru_cache(maxsize=None)
def get_func_name(
    input_shape: tuple,
    kernel_shape: tuple,
    dtype: str,
    mode: str,
    bias: bool,
    stride: tuple,
    padding: tuple,
    dilation: tuple,
    groups: int,
    input_layout: str,
    kernel_layout: str,
    output_layout: str,
) -> str:
    num_spatial_dims = len(input_shape) - 2
    name_items = [
        "conv",
        f"{num_spatial_dims}d",
        str(dtype).removeprefix("torch."),
        str(mode).lower(),
    ]
    if bias and mode == "FORWARD":
        name_items.append("b")
    l2s = lambda l: "x".join([str(i) for i in l])
    name_items.extend(
        [
            l2s(input_shape),
            input_layout.lower(),
            l2s(kernel_shape),
            kernel_layout.lower().replace("n", "f"),
            output_layout.lower().replace("c", "f"),
            l2s(stride) + "s",
            l2s(padding) + "p",
            l2s(dilation) + "d",
            f"{groups}g",
        ]
    )
    return "_".join(name_items)


# Forward Convolution Implementations #

define_schema(
    "convolution",
    "(Tensor x, Tensor w, Tensor? b, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout) -> Tensor",
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


@register_meta("convolution")
def _boo_convolution_meta(
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
    "convolution_backward",
    "(Tensor x, Tensor w, Tensor grad_output, int[] stride, int[] padding, int[] dilation, int groups, str input_layout, str kernel_layout, str output_layout, bool[] mask) -> (Tensor?, Tensor?, Tensor?)",
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


@register_meta("convolution_backward")
def _boo_convolution_backward_meta(
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

        return torch.ops.boo.convolution(
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
        if not BOO_USE_BACKWARD_KERNELS:
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


def boo_convolution(
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
            input_layout,
            kernel_layout,
            output_layout,
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
            input_layout,
            kernel_layout,
            output_layout,
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
        make_tuple(stride, num_spatial_dims),
        make_tuple(padding, num_spatial_dims),
        make_tuple(dilation, num_spatial_dims),
        groups,
        _infer(input_layout),
        _infer(kernel_layout),
        _infer(output_layout),
    )
