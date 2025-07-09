# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from .._support.indexing import *
from ..ops.wave_ops import *
from ..lang.global_symbols import *
from .constraints import *
from .utils.symbol_utils import *


def reorder_workgroups(graph: CapturedTrace, reordering_constraints):
    if len(reordering_constraints) == 0:
        return
    wg_subs = {}
    for c in reordering_constraints:
        wg_subs[c.wg_dim] = c.reordered_equation

    graph_nodes = graph.walk()
    ops_with_implicit_index = (Iterate, GetResult)
    for node in graph_nodes:
        custom_node = get_custom(node)
        if custom_node.index and not isinstance(custom_node, ops_with_implicit_index):
            index = copy.deepcopy(node.index)
            for dim in index:
                index[dim] = index[dim].subs(wg_subs, simultaneous=True)
            node.index = index
