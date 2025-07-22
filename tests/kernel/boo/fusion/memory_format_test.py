# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Testing memory format handling utils.
"""

import pytest

from itertools import permutations
from typing import Sequence

import torch
from iree.turbine.kernel.boo.ops.utils import get_memory_format_permutation

perms = list(permutations(range(4), 4))


@pytest.mark.parametrize("perm", perms)
def test_memory_formats(perm: Sequence[int]):
    x = torch.empty([4, 5, 6, 7])
    x_p = x.permute(perm)
    mem_format_perms = get_memory_format_permutation(x_p)
    if mem_format_perms:
        x_pp = x_p.permute(mem_format_perms.permutation)
        assert (
            x_pp.is_contiguous()
        ), f"Expected contiguous tensor. Got strides {x_pp.stride()}, shape {x_pp.shape}, for perm {perm} and mem_format_perms {mem_format_perms}."
        assert x_pp.shape == x.shape
    else:
        assert x_p.is_contiguous()
    assert list(perm) == [0, 1, 2, 3] or mem_format_perms.inverse_permutation == list(
        perm
    )


@pytest.mark.parametrize("perm", perms)
def test_memory_formats_unit_dim(perm: Sequence[int]):
    x = torch.empty([4, 5, 1, 7])
    x_p = x.permute(perm)
    mem_format_perms = get_memory_format_permutation(x_p)
    if mem_format_perms:
        x_pp = x_p.permute(mem_format_perms.permutation)
        assert (
            x_pp.is_contiguous()
        ), f"Expected contiguous tensor. Got strides {x_pp.stride()}, shape {x_pp.shape}, for perm {perm} and mem_format_perms {mem_format_perms}."
    else:
        assert x_p.is_contiguous()
    if x_p.is_contiguous(memory_format=torch.channels_last):
        assert mem_format_perms.permutation == [
            0,
            2,
            3,
            1,
        ], "Expected channels-last permutation for channels-last tensor."
