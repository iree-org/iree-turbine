# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

import torch
import torch.nn as nn

from iree.turbine import aot
import iree.turbine.ops._jinja_test_ops as ops


class CustomOptionalReturns(nn.Module):
    def __init__(self, mask: tuple[bool, bool]):
        super().__init__()
        self.mask = mask

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        outs = ops.test_optional_returns(a, b, self.mask)
        return tuple([o for o in outs if o is not None])


@pytest.mark.parametrize(
    "mask",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_aot_middle_optional_with_bias(mask):
    a = torch.randn([2, 3, 4])
    b = torch.randn([5])
    e = aot.export(CustomOptionalReturns(mask), args=(a, b))
    e.mlir_module.verify()


@pytest.mark.parametrize(
    "mask",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_eager_middle_optional_with_bias(mask):
    a = torch.randn([2, 3, 4])
    b = torch.randn([5])
    results = ops.test_optional_returns(a, b, mask)
    assert len(results) == 2, "Expected two results regardless of mask."
    assert results[0] is None if not mask[0] else torch.allclose(results[0], a)
    assert results[1] is None if not mask[1] else torch.allclose(results[1], b)
