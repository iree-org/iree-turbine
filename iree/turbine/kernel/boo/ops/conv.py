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
    def forward(ctx, x, w, b=None, kwargs={}):
        x = x.detach()
        w = w.detach()
        ctx.save_for_backward(x, w)
        ctx.kwargs = kwargs
        ctx.use_bias = b is not None
        kwargs["mode"] = "fwd"
        sig = ConvSignature.get(x, w, b, **kwargs)
        ctx.output_layout = sig.output_layout
        conv = get_launchable(sig)
        args = (x, w) if b is None else (x, w, b.detach())
        return conv(*args)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input_grad = weight_grad = bias_grad = None
        x, w = ctx.saved_tensors
        args = (x, w)
        kwargs = ctx.kwargs

        if ctx.needs_input_grad[0]:
            kwargs["mode"] = "bwd"
            bwd_sig = ConvSignature.get(*args, **kwargs)
            bwd_conv = get_launchable(bwd_sig)
            input_grad = bwd_conv(grad_output, w)

        if ctx.needs_input_grad[1]:
            kwargs["mode"] = "wrw"
            wrw_conv = get_launchable(ConvSignature.get(*args, **kwargs))
            weight_grad = wrw_conv(grad_output, x)

        if ctx.needs_input_grad[2] and ctx.use_bias:
            # TODO: use iree to perform the reduce sum?
            output_layout = ctx.output_layout
            reduce_dims = []
            for i, char in enumerate(output_layout):
                if char != "C":
                    reduce_dims.append(i)
            bias_grad = torch.sum(grad_output, reduce_dims)

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
