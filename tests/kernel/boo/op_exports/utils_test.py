# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from iree.turbine.kernel.boo.op_exports.utils import permute_layout


def test_permute_layout():
    assert permute_layout(torch.empty((2, 3, 4, 5)), [0, 2, 3, 1]).is_contiguous(
        memory_format=torch.channels_last
    )
    assert permute_layout(torch.empty((2, 3, 4, 5, 6)), [0, 2, 3, 4, 1]).is_contiguous(
        memory_format=torch.channels_last_3d
    )

    permuted_1 = permute_layout(torch.empty((2, 3, 4)), [0, 2, 1])
    assert list(permuted_1.shape) == [2, 3, 4]
    assert permuted_1.stride() == (12, 1, 3)
