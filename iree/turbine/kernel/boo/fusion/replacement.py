from typing import Callable

import torch

from ..op_exports.conv import DEFAULT_LAYOUTS
from ..ops.utils import get_memory_format_permutation
from torch._decomp import get_decompositions

DecompositionTable = dict[torch._ops.OperatorBase, Callable]


def _convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    _output_padding: tuple[int, ...],
    groups: int,
):
    """Replaces torch.ops.aten.convolution with custom BOO implementation."""
    assert transposed is False, "Decomposition invalid for transposed conv."
    num_spatial_dims = len(input.shape) - 2
    pytorch_layout = DEFAULT_LAYOUTS[num_spatial_dims]

    x_perms = get_memory_format_permutation(input, num_spatial_dims)
    w_perms = get_memory_format_permutation(weight, num_spatial_dims)
    # If weight is non-contiguous, propagate layout from weight.
    # Otherwise, propagate the layout from input.
    output_perms = w_perms or x_perms

    x_contig = input if x_perms is None else input.permute(x_perms.permutation)
    w_contig = weight if w_perms is None else weight.permute(w_perms.permutation)

    to_layout = lambda perms: (
        pytorch_layout
        if perms is None
        else "".join([pytorch_layout[p] for p in perms.permutation])
    )

    call_args = (
        x_contig,
        w_contig,
        bias,
        stride,
        padding,
        dilation,
        groups,
        to_layout(x_perms),
        to_layout(w_perms),
        to_layout(output_perms),
    )
    replacement_conv = torch.ops.boo.layout_customizable_convolution.default(*call_args)
    return (
        replacement_conv
        if output_perms is None
        else replacement_conv.permute(output_perms.inverse_permutation)
    )


def _convolution_backward(
    grad_output,
    x,
    w,
    _bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    _output_padding,
    groups,
    output_mask,
):
    "Replace 'torch.ops.aten.convolution_backward' with custom BOO implementation."
    assert transposed is False, "Decomposition invalid for transposed backward conv."

    num_spatial_dims = len(grad_output.shape) - 2
    pytorch_layout = DEFAULT_LAYOUTS[num_spatial_dims]

    x_perms = get_memory_format_permutation(x, num_spatial_dims)
    w_perms = get_memory_format_permutation(w, num_spatial_dims)
    output_perms = get_memory_format_permutation(grad_output, num_spatial_dims)

    x_contig = x if x is None or x_perms is None else x.permute(x_perms.permutation)
    w_contig = w if w is None or w_perms is None else w.permute(w_perms.permutation)
    grad_output_contig = (
        grad_output
        if grad_output is None or output_perms is None
        else grad_output.permute(output_perms.permutation)
    )

    to_layout = lambda perms: (
        pytorch_layout
        if perms is None
        else "".join([pytorch_layout[p] for p in perms.permutation])
    )

    call_args = (
        grad_output_contig,
        x_contig,
        w_contig,
        stride,
        padding,
        dilation,
        groups,
        to_layout(x_perms),
        to_layout(w_perms),
        to_layout(output_perms),
        output_mask,
    )
    ret_grad_perms = (x_perms, w_perms, None)
    grads = torch.ops.boo.layout_customizable_convolution_backward.default(*call_args)
    return tuple(
        g if g is None or _perm is None else g.permute(_perm.inverse_permutation)
        for (g, _perm) in zip(grads, ret_grad_perms, strict=True)
    )


# We avoid direct registry of these decompositions with pytorch in case conflicts could arise.
DEFAULT_BOO_OP_DECOMPOSITION_TABLE: DecompositionTable = {
    # Custom Op implementations.
    torch.ops.aten.convolution.default: _convolution,
    torch.ops.aten.convolution_backward.default: _convolution_backward,
}

DEFAULT_PYTORCH_OP_DECOMPOSE_LIST = [
    # Norm ops with existing torch decompositions.
    torch.ops.aten.native_layer_norm_backward,
    torch.ops.aten.native_group_norm,
    torch.ops.aten.native_layer_norm,
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    # This is included in one of the above decompositions and not handled in torch-mlir.
    torch.ops.aten.squeeze.dims,
]

DEFAULT_BOO_OP_DECOMPOSITIONS: DecompositionTable = (
    get_decompositions(DEFAULT_PYTORCH_OP_DECOMPOSE_LIST)
    | DEFAULT_BOO_OP_DECOMPOSITION_TABLE
)
