# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Callable
import unittest
import os
import pytest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph, add_get_results
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.ops.wave_ops import get_custom
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.visualization import visualize_graph
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.wave.utils.graph_utils import initialize_iter_args


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()
    return func


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K

# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

# Induction variable for dimension K
ARGK = tkl.sym.ARGK


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.iterate(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


graphviz_disabled = False
try:
    import pygraphviz
except:
    graphviz_disabled = True


@pytest.mark.xfail(condition=graphviz_disabled, reason="pygraphviz not installed")
@run
def test_gemm():
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K, ARGK)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2, THREAD_0 / 64)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2, THREAD_1)]
    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]
    with tk.gen.TestLaunchContext(
        {
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
        }
    ):
        graph = gemm()
        IndexingContext.current().finalize()
        initialize_iter_args(graph)
        add_get_results(graph)
        infer_types(graph)
        set_node_indices(graph, constraints)
        expand_graph(graph, constraints)
        set_post_expansion_indices(graph, constraints)
        visualize_graph(graph.get_subgraph("region_0"), "gemm.png")
        assert os.path.exists("gemm.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
