# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from .utils import perf_test

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

# Add performance test markers.
for test_name in _e2e_test_shapes:
    _e2e_test_shapes[test_name] += [perf_test(x) for x in _e2e_test_shapes[test_name]]


def get_test_shapes(test_name: str):
    assert test_name in _e2e_test_shapes, f"Unknown test name: {test_name}"
    return _e2e_test_shapes[test_name]
