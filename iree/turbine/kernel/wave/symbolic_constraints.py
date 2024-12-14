# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.turbine.kernel._support.indexing import IndexExpr, IndexSymbol
from dataclasses import dataclass
from typing import Callable


@dataclass
class SymbolicAlias:
    """
    A constraint of the form `tkw.SymbolicConstraint(K, SYMBOLIC_K)` specifies
    that the relationship between the source and target symbols is given by
    source = source_to_target(target).

    SymbolicAliases are modeled in the compiler as additional workgroup, wave,
    and tiling constraints that are derived from the source. They are ignored
    during expansion and utilize the same workgroup and wave ids as the
    target symbol.
    """

    source: IndexSymbol | IndexExpr
    target: IndexSymbol | IndexExpr
    source_to_target: Callable[[IndexSymbol | IndexExpr], IndexSymbol | IndexExpr]
