# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch

from ..conv_exports import (
    ConvSignature,
    DEFAULT_LAYOUTS,
)

from ..ops.utils import (
    CHANNELS_LAST_LAYOUTS,
    CHANNELS_LAST_MEMORY_FORMAT,
    CHANNELS_LAST_TO_CONTIGUOUS_PERMUTATION,
    CONTIGUOUS_TO_CHANNELS_LAST_PERMUTATION,
)


def replace_convolution_node(conv_node: torch.fx.Node) -> None:
    if (
        conv_node.op != "call_function"
        or conv_node.target != torch.ops.aten.convolution.default
    ):
        raise ValueError(
            f"Expected conv_node to be torch.ops.aten.convolution.default. Got {conv_node}."
        )
    (
        x_node,
        w_node,
        maybe_b_node,
        stride,
        dilation,
        padding,
        transposed,
        output_padding,
        groups,
    ) = conv_node.args
    if transposed:
        return
    x = x_node.meta.get("val")
    assert isinstance(
        x, torch.Tensor
    ), f"Expected fake tensor to be present for node {x_node}."
    w = w_node.meta.get("val")
    assert isinstance(
        w, torch.Tensor
    ), f"Expected fake tensor to be present for node {w_node}."
    bias = maybe_b_node is not None

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
    output_layout = kernel_layout

    input_shape = tuple([x.shape[i] for i in cl_contig_perm]) if x_cl else list(x.shape)
    kernel_shape = (
        tuple([w.shape[i] for i in cl_contig_perm]) if w_cl else list(w.shape)
    )
    signature = ConvSignature(
        input_shape=input_shape,
        kernel_shape=kernel_shape,
        input_layout=input_layout,
        kernel_layout=kernel_layout,
        output_layout=output_layout,
        bias=bias,
        dtype=x.dtype,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    nn_module = signature.get_nn_module(use_custom=True)

    def boo_convolution_replacement(
        x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
    ) -> torch.Tensor:
        x = x if not x_cl else x.permute(cl_contig_perm)
        w = w if not w_cl else w.permute(cl_contig_perm)
        args = (x, w, b) if bias else (x, w)
        output = nn_module.forward(*args)
        return output.permute(contig_cl_perm) if w_cl else output

    new_args = (x_node, w_node, maybe_b_node)

    conv_node.target = boo_convolution_replacement
    conv_node.args = new_args
