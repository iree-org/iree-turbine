# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch


def test_basic():
    def foo(x, y):
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

    opt_foo1 = torch.compile(foo, backend="turbine_cpu")
    print(opt_foo1(torch.randn(10, 10), torch.randn(10, 10)))
