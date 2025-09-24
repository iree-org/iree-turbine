# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import Sequence, Tuple

import torch

from ..op_exports.conv import (
    ConvSignature,
    DEFAULT_LAYOUTS,
    get_conv_func_name,
)
from ..driver.launch import get_launchable

from ..runtime import LaunchableRuntimeCache

from .library import register_meta, BOO_LIBRARY
from .utils import *
from .layout_customizable_conv import boo_layout_customizable_convolution

from ....runtime.op_reg import CustomOp, KernelSelection, KernelBuilder

__all__ = [
    "boo_conv",
    "boo_convolution",
]

# Forward Convolution Implementations #


@CustomOp.register(library=BOO_LIBRARY)
class convolution(CustomOp):
    signature = "convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor"

    def select(self, ksel: KernelSelection):
        # Declare args.
        x = ksel.arg_tensor(0)
        w = ksel.arg_tensor(1)
        b = ksel.arg_optional_tensor(2)
        stride = ksel.attr_list_int(3)
        padding = ksel.attr_list_int(4)
        dilation = ksel.attr_list_int(5)
        groups = ksel.attr_int(6)
        # Specialize args.
        x.specialize_all_dims()
        w.specialize_all_dims()
        if b:
            b.specialize_all_dims()
            ksel.variant = "biased"

        # Manually set memory_format specialization to correct meta impl for eager_execution.
        num_spatial_dims = len(x.spec_dims) - 2
        cl_mem_format = CHANNELS_LAST_MEMORY_FORMAT.get(num_spatial_dims)
        output_mem_format = (
            cl_mem_format
            if cl_mem_format and w.t.is_contiguous(memory_format=cl_mem_format)
            else torch.contiguous_format
        )
        self.conv_sig = ConvSignature(
            input_shape=x.spec_dims,
            kernel_shape=w.spec_dims,
            bias=(b is not None),
            dtype=x.t.dtype,
            stride=stride.v,
            padding=padding.v,
            dilation=dilation.v,
            groups=groups.v,
        )
        output_shape = self.conv_sig.output_shape
        o_meta = torch.empty(
            tuple(output_shape),
            dtype=self.conv_sig.dtype,
            memory_format=output_mem_format,
            device="meta",
        )
        ksel.return_tensor(o_meta)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        raise NotImplementedError("convolution generate NYI")

    def eager_execute(self, *args):
        return _boo_convolution_impl(*args)


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


# Backward Convolution Implementations #


@CustomOp.register(library=BOO_LIBRARY, register_meta=False)
class convolution_backward(CustomOp):
    signature = "convolution_backward(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[] mask) -> (Tensor?, Tensor?, Tensor?)"

    def select(self, ksel):
        raise NotImplementedError("convolution_backward select NYI")

    def generate(self, ksel, kb):
        raise NotImplementedError("convolution_backward generate NYI")

    def eager_execute(self, *args):
        return _boo_convolution_backward_impl(*args)


def _boo_convolution_backward_impl(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    groups: int,
    mask: Tuple[bool, bool, bool],
) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

    num_spatial_dims = len(x.shape) - 2

    default_layout = DEFAULT_LAYOUTS[num_spatial_dims]

    input_perms = get_memory_format_permutation(x, num_spatial_dims)
    kernel_perms = get_memory_format_permutation(w, num_spatial_dims)
    output_perms = get_memory_format_permutation(grad_output, num_spatial_dims)

    input_layout = (
        default_layout
        if input_perms is None
        else "".join([default_layout[i] for i in input_perms.permutation])
    )
    kernel_layout = (
        default_layout
        if kernel_perms is None
        else "".join([default_layout[i] for i in kernel_perms.permutation])
    )
    output_layout = (
        default_layout
        if output_perms is None
        else "".join([default_layout[i] for i in output_perms.permutation])
    )

    x_contig = x if input_perms is None else x.permute(input_perms.permutation)
    w_contig = w if kernel_perms is None else w.permute(kernel_perms.permutation)
    dLdy_contig = (
        grad_output
        if output_perms is None
        else grad_output.permute(output_perms.permutation)
    )

    sig = ConvSignature(
        input_shape=x_contig.shape,
        kernel_shape=w_contig.shape,
        input_layout=input_layout,
        kernel_layout=kernel_layout,
        output_layout=output_layout,
        dtype=x_contig.dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        backward_mask=mask,
    )

    launch = get_launchable(sig, use_custom=True)

    grads = launch(dLdy_contig.data, x_contig.data, w_contig.data)

    if isinstance(grads, torch.Tensor):
        grads = (grads,)

    outputs = [None, None, None]
    g_idx = 0

    for o_idx, m in enumerate(mask):
        if m:
            outputs[o_idx] = grads[g_idx]
            g_idx += 1

    handle_return = lambda ret, perms: (
        ret if ret is None or perms is None else ret.permute(perms.inverse_permutation)
    )

    return (
        handle_return(outputs[0], input_perms),
        handle_return(outputs[1], kernel_perms),
        outputs[2],
    )


@register_meta("convolution_backward")
def _boo_convolution_backward_meta(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
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
        [w.shape[0]],
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
            grad_output,
            x,
            w,
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
