# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pytest

# List of all test shapes for end to end tests.
_e2e_test_shapes = {}

# Order of shapes: (B, M, N, K1, K2)
_e2e_test_shapes["attention"] = [
    (8, 128, 128, 64, 256),
    (40, 1024, 64, 64, 1024),
]
_e2e_test_shapes["chained_gemm"] = _e2e_test_shapes["attention"]
_e2e_test_shapes["decode_attention"] = _e2e_test_shapes["attention"]

# Order of shapes: (B, BN, K2, H, K1, M, N)
_e2e_test_shapes["evoformer"] = [
    (1, 256, 256, 4, 32, 256, 32),
    (1, 512, 256, 8, 8, 256, 8),
]


def get_test_shapes(test_name: str):
    assert test_name in _e2e_test_shapes, f"Unknown test name: {test_name}"
    shapes = [
        pytest.param(s, id="x".join(map(str, s))) for s in _e2e_test_shapes[test_name]
    ]
    shapes += [
        pytest.param(s, id="x".join(map(str, s)) + "-perf", marks=pytest.mark.perf_only)
        for s in _e2e_test_shapes[test_name]
    ]
    return shapes
