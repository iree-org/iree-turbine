# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from .._support.indexing import IndexExpr


@dataclass
class Assumption:
    """
    Assumptions are sympy assumptions that can be used to
    make decisions during code generation. These can be
    statements such as bounds on sympy variables. For example,
    we can state that

    Assumption(M < 64)

    and then later make queries based on this assumption, such as

    evaluate(M > 70) -> False
    evaluate(M < 32) -> None (because we cannot say one way or the other)
    evaluate(M < 70) -> True

    """

    expr: IndexExpr
