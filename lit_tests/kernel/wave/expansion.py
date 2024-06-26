# RUN: python %s | FileCheck %s

import logging
from typing import Callable
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel.ops.wave_ops import get_custom


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()

    return func


def print_trace(trace: CapturedTrace):
    """
    Prints all subgraphs of a trace starting with the root graph.
    The graphs are printed first in the torch printing format and then using
    our custom node format.
    """
    # The root graph is at the back so we print the subgraphs in reverse order
    for subgraph in reversed(list(trace.region_graph.subgraphs.values())):
        print(subgraph)
        for node in subgraph.nodes:
            print(get_custom(node))


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K

# Workgroup tile sizes
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_L

# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


@tkw.wave_trace_only()
def read_write(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


def test_read_write():
    with tk.gen.TestLaunchContext({}):
        graph = read_write()
        expand_graph(graph)
        print_trace(graph)


@tkw.wave_trace_only()
def gemm(
    a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    c_reg = tkl.Register[M, N, tkl.f32](0.0)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
        a_reg = tkw.read(a, elements_per_thread=4)
        b_reg = tkw.read(b, elements_per_thread=4)
        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, c, elements_per_thread=4)


@run
def test_gemm():
    with tk.gen.TestLaunchContext({}):
        graph = gemm()
        expand_graph(graph)
        print_trace(graph)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
