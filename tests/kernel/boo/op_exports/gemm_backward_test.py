# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List
import pytest

import torch
from iree.turbine.kernel.boo.op_exports.gemm import (
    Mode,
    GEMMSignature,
)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("shapes", [([32, 64], [64, 12]), ([12, 16], [16, 24])])
@pytest.mark.parametrize(
    "transpose", [(True, True), (False, True), (True, False), (True, True)]
)
def test_gemm_backward_impl(
    dtype: torch.dtype,
    shapes: tuple[List[int]],
    transpose: tuple[List[int]],
):
    """Tests our custom implementation of gradients using PyTorch operations
    against PyTorch gradients."""

    kwargs = {
        "a_shape": shapes[0],
        "b_shape": shapes[1],
        "transpose_a": transpose[0],
        "transpose_b": transpose[1],
        "dtype": dtype,
    }
    fwd_sig = GEMMSignature(**kwargs)
    args = fwd_sig.get_sample_args(seed=1)

    args = tuple(arg.to(device="cpu").requires_grad_(True) for arg in args)
    fwd = fwd_sig.get_nn_module().to(device="cpu")
    bwd_a_sig = GEMMSignature(**kwargs, mode=Mode.A_BACKWARD)
    bwd_a = bwd_a_sig.get_nn_module().to(device="cpu")
    bwd_b_sig = GEMMSignature(**kwargs, mode=Mode.B_BACKWARD)
    bwd_b = bwd_b_sig.get_nn_module().to(device="cpu")

    # Forward pass
    fwd_results = fwd(*args)
    fwd_results.retain_grad()
    loss = fwd_results.sum()
    loss.backward(retain_graph=True)

    # Get gradient of output
    dLoss = fwd_results.grad

    # Test backward for A
    bwd_a_args = bwd_a_sig.arrange_backward_launch_args(args, fwd_results)
    dLossDa = bwd_a(dLoss, *bwd_a_args)

    # Test backward for B
    bwd_b_args = bwd_b_sig.arrange_backward_launch_args(args, fwd_results)
    dLossDb = bwd_b(dLoss, *bwd_b_args)

    grads = [dLossDa, dLossDb]

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


if __name__ == "__main__":
    pytest.main(["-s", __file__])
