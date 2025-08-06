# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import pytest

import torch
from iree.turbine.kernel.boo.op_exports.layer_norm import (
    Mode,
    LayerNormSignature,
)


def _marked_xfail(*args):
    return pytest.param(
        *args,
        marks=pytest.mark.xfail(
            condition=not torch.cuda.is_available(),
            reason="Cannot run on GPU with no GPU.",
        ),
    )


def _get_results(
    input_shape: Sequence[int],
    normalized_shape: Sequence[int],
    elementwise_affine: bool,
    bias: bool,
    dtype: torch.dtype,
    mode: Mode,
    device: torch.device,
    use_aten: bool,
) -> tuple[torch.Tensor, ...]:
    signature = LayerNormSignature(
        input_shape=input_shape,
        normalized_shape=normalized_shape,
        elementwise_affine=elementwise_affine,
        bias=bias,
        use_aten=use_aten,
        dtype=dtype,
        mode=mode,
    )
    module = signature.get_nn_module(use_custom=True).to(device=device)
    args = signature.get_sample_args(device=device, seed=42)
    results = module(*args)
    return results if isinstance(results, tuple) else (results,)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "elementwise_affine_bias", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("device", ["cpu", _marked_xfail("cuda")])
@pytest.mark.parametrize("mode", list(Mode.__members__.keys()))
def test_layer_norm_impl(
    dtype: torch.dtype,
    elementwise_affine_bias: tuple[bool, bool],
    device: str,
    mode: Mode,
):
    """Tests our custom implementation of forward and backward operations against ATen."""

    elementwise_affine, bias = elementwise_affine_bias
    input_shape = [4, 8, 16, 32]
    normalized_shape = [16, 32]
    aten = _get_results(
        input_shape,
        normalized_shape,
        elementwise_affine,
        bias,
        dtype,
        mode,
        device,
        True,
    )
    manual = _get_results(
        input_shape,
        normalized_shape,
        elementwise_affine,
        bias,
        dtype,
        mode,
        device,
        False,
    )

    atol = 1e-5
    rtol = 1e-5
    assert len(aten) == len(manual)
    for a, m in zip(aten, manual):
        assert (a is None and m is None) or (a is not None and m is not None)
        if a is None:
            continue
        assert torch.allclose(a, m, atol=atol, rtol=rtol)
