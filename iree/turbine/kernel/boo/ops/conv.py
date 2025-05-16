# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from ..conv_exports import ConvSignature, get_launchable

__all__ = [
    "boo_conv",
]


class _BooConv(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, w, b=None, kwargs={}):
        x = x.detach()
        w = w.detach()
        device_type = x.device.type

        do_autocast = device_type == "cuda" and torch.is_autocast_enabled("cuda")

        # set original dtype and autocast dtype
        _og_dtype = x.dtype
        _dtype = w.dtype if not do_autocast else torch.get_autocast_dtype("cuda")

        # convert everything to w.dtype or autocast dtype
        x = x.to(dtype=_dtype)
        w = w.to(dtype=_dtype)
        b = b if b is None else b.to(dtype=_dtype)

        # save lowp inputs for backward kernels
        ctx.save_for_backward(x, w)

        ctx.kwargs = kwargs
        ctx.use_bias = b is not None
        kwargs["mode"] = "fwd"

        # TODO: set the dtype such that we don't cast back down to bfloat16 if autocasted from f32.
        sig = ConvSignature.get(x, w, b, **kwargs)

        # Save the output layout for backward conv.
        ctx.output_layout = sig.output_layout

        # Get a launchable and apply.
        conv = get_launchable(sig)
        args = (x, w) if b is None else (x, w, b.detach())
        result = conv(*args)

        # Cast back to original dtype if necessary.
        return result.to(dtype=_og_dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input_grad = weight_grad = bias_grad = None
        x, w = ctx.saved_tensors
        args = (x, w)
        kwargs = ctx.kwargs

        device_type = grad_output.device.type
        do_autocast = device_type == "cuda" and torch.is_autocast_enabled("cuda")

        # get original and autocast dtypes
        _og_dtype = grad_output.dtype
        _dtype = _og_dtype if not do_autocast else torch.get_autocast_dtype("cuda")

        # saved tensors are already lowp, so just cast grad_output
        grad_output = grad_output.to(dtype=_dtype)

        if ctx.needs_input_grad[0]:
            kwargs["mode"] = "bwd"
            bwd_sig = ConvSignature.get(*args, **kwargs)
            bwd_conv = get_launchable(bwd_sig)
            input_grad = bwd_conv(grad_output, w)
            input_grad = input_grad.to(dtype=_og_dtype)

        if ctx.needs_input_grad[1]:
            kwargs["mode"] = "wrw"
            wrw_conv = get_launchable(ConvSignature.get(*args, **kwargs))
            weight_grad = wrw_conv(grad_output, x)
            weight_grad = weight_grad.to(dtype=_og_dtype)

        if ctx.needs_input_grad[2] and ctx.use_bias:
            # TODO: use iree to perform the reduce sum?
            output_layout = ctx.output_layout
            reduce_dims = []
            for i, char in enumerate(output_layout):
                if char != "C":
                    reduce_dims.append(i)
            bias_grad = torch.sum(grad_output, reduce_dims)
            bias_grad = bias_grad.to(dtype=_og_dtype)

        # return `None` for kwargs dict backward, since this will never be needed.
        return input_grad, weight_grad, bias_grad, None


def boo_conv(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    **kwargs
):
    """
    Applies a differentiable forward convolution kernel.

    kwargs can include any of the following usual convolution options:

        stride         : int or int[]
        padding        : int or int[]
        dilation       : int or int[]
        groups         : int
        output_padding : int or int[]
        transposed     : bool

    Users can also specify alternative layouts for each convolution:

        shared_layout  : str
        input_layout   : str
        kernel_layout  : str
        output_layout  : str

    These layouts should be permutations of "NCH", "NCHW", or "NCDHW".
    """

    filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return _BooConv.apply(input, weight, bias, filtered_kwargs)
