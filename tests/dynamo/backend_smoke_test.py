# Copyright 2023 Nod Labs, Inc
# SPDX-FileCopyrightText: 2024 The IREE Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


def test_basic():
    def foo(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    opt_foo1 = torch.compile(foo, backend="turbine_cpu")
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
