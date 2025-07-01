# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from .library import define_schema, register_impl, register_meta
from ..layer_norm_exports.layer_norm import LayerNormSignature, Mode
from ..driver.launch import get_launchable
from ..runtime import LaunchableRuntimeCache
from .utils import *
from typing import Sequence

__all__ = [
    "boo_layer_norm",
]

# TODO(azinenko): can this be automated, pytorch doc says these can be inferred from type information?
define_schema(
    "layer_norm",
    "(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float? eps) -> (Tensor, Tensor, Tensor)",
)
# "(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-5) -> Tensor")

# TODO(azinenko,zjgarvey): this should eventually be generalized with non-boo registration.


@register_impl("layer_norm")
def _boo_layer_norm_impl(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    signature = LayerNormSignature.get(input, normalized_shape, weight, bias, eps=eps)

    # TODO: support non-contiguous memory formats via permutations

    func_name = signature.get_func_name()
    args = tuple(
        filter(
            lambda x: x is not None,
            map(lambda x: x.data if x is not None else None, (input, weight, bias)),
        )
    )
    cache_hit = LaunchableRuntimeCache.get(func_name)
    if cache_hit:
        return cache_hit(*args)

    layer_norm = get_launchable(signature)
    return layer_norm(*args)


@register_meta("layer_norm")
def _boo_layer_norm_meta(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    signature = LayerNormSignature.get(input, normalized_shape, weight, bias, eps=eps)

    # TODO: support non-contiguous memory formats via permutations

    return (
        torch.empty_like(input),
        torch.empty(
            signature.aggregate_shape, dtype=signature.dtype, device=input.device
        ),
        torch.empty(
            signature.aggregate_shape, dtype=signature.dtype, device=input.device
        ),
    )


define_schema(
    "layer_norm_backward",
    "(Tensor grad_output, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor weight, Tensor bias, bool[3] mask) -> (Tensor?, Tensor?, Tensor?)",
)


@register_impl("layer_norm_backward")
def _boo_layer_norm_backward_impl(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: int | Sequence[int] | torch.Size,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: tuple[bool, bool, bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

    input_grad: torch.Tensor | None = None
    weight_grad: torch.Tensor | None = None
    bias_grad: torch.Tensor | None = None

    # TODO(azinenko): it is unclear to me why convolution decided to implement
    # each derivative computation as a single kernel, but cargo-culting it here.

    def data_tuple(*args: torch.Tensor):
        return tuple(a.data for a in args)

    if mask[0]:
        signature = LayerNormSignature.get(
            input, normalized_shape, weight, bias, Mode.INPUT_BACKWARD
        )
        launchable = get_launchable(signature)
        input_grad = launchable(*data_tuple(grad_output, input, weight, mean, rstd))

    if mask[1]:
        signature = LayerNormSignature.get(
            input, normalized_shape, weight, bias, Mode.WEIGHT_BACKWARD
        )
        launchable = get_launchable(signature)
        weight_grad = launchable(*data_tuple(grad_output, input, mean, rstd))

    if mask[2]:
        signature = LayerNormSignature.get(
            input, normalized_shape, weight, bias, Mode.BIAS_BACKWARD
        )
        launchable = get_launchable(signature)
        bias_grad = launchable(*data_tuple(grad_output))

    return input_grad, weight_grad, bias_grad


@register_meta("layer_norm_backward")
def _boo_layer_norm_backward_meta(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: int | Sequence[int] | torch.Size,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: tuple[bool, bool, bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    input_grad: torch.Tensor | None = None
    weight_grad: torch.Tensor | None = None
    bias_grad: torch.Tensor | None = None

    if mask[0]:
        input_grad = torch.empty_like(input)
    if mask[1]:
        weight_grad = torch.empty_like(weight)
    if mask[2]:
        bias_grad = torch.empty_like(bias)
    return input_grad, weight_grad, bias_grad


def pytorch_layer_norm_backward(ctx, grad_output: torch.Tensor):
    """ATen/PyTorch fallback implementation for backward."""

    input, weight, bias, mean, rstd = ctx.saved_tensors
    mask = tuple(ctx.needs_input_grad[0:3])

    input_grad, weight_grad, bias_grad = torch.ops.aten.native_layer_norm_backward(
        grad_output, input, ctx.normalized_shape, mean, rstd, weight, bias, mask
    )

    return input_grad, None, weight_grad, bias_grad, None


class _BooLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        normalized_shape: int | Sequence[int] | torch.Size,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        result, mean, rstd = torch.ops.boo.layer_norm(
            input, normalized_shape, weight, bias, eps
        )
        ctx.save_for_backward(input, weight, bias, mean, rstd)
        ctx.normalized_shape = normalized_shape
        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[
        torch.Tensor | None, None, torch.Tensor | None, torch.Tensor | None, None
    ]:
        if not is_boo_backward_enabled():
            return pytorch_layer_norm_backward(ctx, grad_output)

        input, weight, bias, mean, rstd = ctx.saved_tensors
        # Note that the context contains grad flags for every forward argument
        # in order, including non-differentiable attributes like
        # `normalized_shape`. The indices below correspond to the positions of
        # input, weight and bias in the forward signature.
        mask = (
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[2],
            ctx.needs_input_grad[3],
        )
        input_grad, weight_grad, bias_grad = torch.ops.boo.layer_norm_backward(
            grad_output, input, ctx.normalized_shape, mean, rstd, weight, bias, mask
        )

        return input_grad, None, weight_grad, bias_grad, None


def boo_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    use_autograd = torch._C.is_grad_enabled() and any(
        x is not None and x.requires_grad for x in (input, weight, bias)
    )
    if use_autograd:
        return _BooLayerNorm.apply(input, normalized_shape, weight, bias, eps)
    result, _, _ = torch.ops.boo.layer_norm(input, normalized_shape, weight, bias, eps)
    return result
