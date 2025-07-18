# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import torch
from iree.turbine.kernel.boo.layer_norm_exports.layer_norm import (
    Mode,
    LayerNormSignature,
)


# Note that elementwise_affine and bias flags are grouped together to avoid an
# invalid combination.
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("input_shape", [(10, 12, 14, 16), (11, 13, 15)])
@pytest.mark.parametrize(
    "elementwise_affine_bias", [(False, False), (True, False), (True, True)]
)
def test_layer_norm_impl(
    dtype: torch.dtype,
    input_shape: tuple[int, ...],
    elementwise_affine_bias: tuple[bool, bool],
):
    """Tests our custom implementation of gradients using PyTorch operations
    against PyTorch gradients."""
    elementwise_affine, bias = elementwise_affine_bias

    # TODO: consider generalizing the get_fwd_signature method to return any
    # other mode.
    kwargs = {
        "input_shape": input_shape,
        "normalized_shape": input_shape[-1:],
        "elementwise_affine": elementwise_affine,
        "bias": bias,
        "dtype": dtype,
    }
    fwd_sig = LayerNormSignature(**kwargs)
    args = fwd_sig.get_sample_args(seed=1)

    # TODO(azinenko): cargo-culted the device="cpu" bit here, unclear why we are
    # not testing on GPU.
    args = tuple(arg.to(device="cpu").requires_grad_(True) for arg in args)
    fwd = fwd_sig.get_nn_module(use_custom=True).to(device="cpu")
    bwd_input_sig = LayerNormSignature(**kwargs, mode=Mode.INPUT_BACKWARD)
    bwd_input = bwd_input_sig.get_nn_module(use_custom=True).to(device="cpu")
    bwd_weight_sig = LayerNormSignature(**kwargs, mode=Mode.WEIGHT_BACKWARD)
    bwd_weight = bwd_weight_sig.get_nn_module(use_custom=True).to(device="cpu")
    bwd_bias_sig = LayerNormSignature(**kwargs, mode=Mode.BIAS_BACKWARD)
    bwd_bias = bwd_bias_sig.get_nn_module(use_custom=True).to(device="cpu")

    fwd_results = fwd(*args)
    main_result = fwd_results[fwd_sig.main_result_index]
    main_result.retain_grad()
    # TODO: this is not a good loss function (#1021).
    loss = main_result.sum()
    loss.backward(retain_graph=True)

    dLossDOutput = main_result.grad
    bwd_input_args = bwd_input_sig.arrange_backward_launch_args(args, fwd_results)
    dLossDInput = bwd_input(dLossDOutput, *bwd_input_args)
    grads = [dLossDInput]

    if elementwise_affine:
        bwd_weight_args = bwd_weight_sig.arrange_backward_launch_args(args, fwd_results)
        dLossDWeights = bwd_weight(dLossDOutput, *bwd_weight_args)
        grads.append(dLossDWeights)

    if bias:
        bwd_bias_args = bwd_bias_sig.arrange_backward_launch_args(args, fwd_results)
        dLossDBias = bwd_bias(dLossDOutput, *bwd_bias_args)
        grads.append(dLossDBias)

    rtol = 1e-4
    atol = 1e-4
    assert len(grads) == len(args)
    results = [
        torch.allclose(arg.grad, grad, rtol=rtol, atol=atol)
        for arg, grad in zip(args, grads)
    ]
    if all(results):
        return

    for i, r in enumerate(results):
        if r:
            continue
        print(f"Expected for gradient #{i}: ", args[i].grad)
        print(f"Actual for gradient #{i}: ", grads[i])
    raise RuntimeError(f"Tensor matches: {results}")
