# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from .library import define_schema, register_impl, register_meta
from ..layer_norm_exports.layer_norm import LayerNormSignature
from ..layer_norm_exports.launch import get_launchable
from typing import Sequence

__all__ = [
    "boo_layer_norm",
]

# TODO(azinenko): can this be automated, pytorch doc says these can be inferred from type information?
define_schema(
    "layer_norm",
    "(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float? eps) -> Tensor",
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
) -> torch.Tensor:
    signature = LayerNormSignature.get(input, normalized_shape, weight, bias, eps=eps)

    # TODO: caching
    # func_name = signature.get_func_name()

    layer_norm = get_launchable(signature)
    args = tuple(
        filter(
            lambda x: x is not None,
            map(lambda x: x.data if x is not None else None, (input, weight, bias)),
        )
    )
    # TODO: check if we don't need to pass the normalized shape and eps somehow
    return layer_norm(*args)


@register_meta("layer_norm")
def _boo_layer_norm_meta(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # TODO(azinenko): this is rather unfortunate to have this duplicated with impl
    signature = LayerNormSignature.get(input, normalized_shape, weight, bias, eps=eps)

    # TODO: do we care about memory format here? It is probably the same as input

    return torch.empty(
        signature.input_shape, dtype=signature.dtype, device=input.device
    )


def boo_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    return torch.ops.boo.layer_norm(input, normalized_shape, weight, bias, eps)
