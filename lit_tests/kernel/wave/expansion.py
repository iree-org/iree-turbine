# RUN: python %s | FileCheck %s

import logging
from typing import Callable
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.constraints import MMAType
from shark_turbine.kernel.wave.expansion import expand_graph
from shark_turbine.kernel._support.tracing import CapturedTrace
from shark_turbine.kernel.ops.wave_ops import get_custom


def run(func: Callable[[], None]) -> Callable[[], None]:
    """Run a function as part of the test suite."""
    if __name__ == "__main__":
        func()
        # Print a separator between tests
        print("-----")
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
def read_write_same_size(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run
def test_read_write():
    with tk.gen.TestLaunchContext({}):
        graph = read_write_same_size()
        expand_graph(graph)
        print_trace(graph)
        # CHECK: %a
        # CHECK: %c
        # CHECK: %read
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_1_1_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_1_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_0_1_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %write_0_0_0
        # CHECK-SAME: (%read, %c, 4)
        # CHECK: %write_1_1_0
        # CHECK-SAME: (%read_1_1_0, %c, 4)
        # CHECK: %write_1_0_0
        # CHECK-SAME: (%read_1_0_0, %c, 4)
        # CHECK: %write_0_1_0
        # CHECK-SAME: (%read_0_1_0, %c, 4)

        # CHECK: -----


@tkw.wave_trace_only()
def read_write_different_dims(
    a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    c: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f32],
):
    a_reg = tkw.read(a, elements_per_thread=4)
    tkw.write(a_reg, c, elements_per_thread=4)


@run
def test_read_write():
    with tk.gen.TestLaunchContext({}):
        graph = read_write_different_dims()
        expand_graph(graph)
        print_trace(graph)
        # CHECK: %a
        # CHECK: %c
        # CHECK: %read_0_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_1_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %write_0_0_0
        # CHECK-SAME: (%read_0_0_0, %c, 4)
        # CHECK: %write_1_0_1
        # CHECK-SAME: (%read_1_0_0, %c, 4)
        # CHECK: %write_1_0_0
        # CHECK-SAME: (%read_1_0_0, %c, 4)
        # CHECK: %write_0_0_1
        # CHECK-SAME: (%read_0_0_0, %c, 4)

        # CHECK: -----


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
        constraints: list[tkw.Constraint] = [
            tkw.HardwareConstraint(MMAType.F32_16x16x16_F16, 64, [1, 1, 1])
        ]
        expand_graph(graph, constraints)
        print_trace(graph)
        # Root graph:
        # CHECK: %a
        # CHECK: %b
        # CHECK: %c
        # CHECK: %register_0_0_0
        # CHECK: %register_1_1_0
        # CHECK: %register_1_0_0
        # CHECK: %register_0_1_0
        # CHECK: %reduction
        # CHECK: %getresult_1_1_0
        # CHECK: %getresult_1_0_0
        # CHECK: %getresult_0_1_0
        # CHECK: %getresult_0_0_0
        # CHECK: %write_0_0_0
        # TODO: This link-up is not yet correct!
        # CHECK-SAME: (%reduction, %c, 4)
        # CHECK: %write_1_1_0
        # CHECK-SAME: (%get_result_1_1_0, %c, 4)
        # CHECK: %write_1_0_0
        # CHECK-SAME: (%get_result_1_0_0, %c, 4)
        # CHECK: %write_0_1_0
        # CHECK-SAME: (%get_result_0_1_0, %c, 4)

        # Reduction subgraph:

        # CHECK: %acc
        # CHECK: %acc_1_1_0
        # CHECK: %acc_1_0_0
        # CHECK: %acc_0_1_0

        # CHECK: %a
        # CHECK: %read_0_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_0_0_1
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_1_0_0
        # CHECK-SAME: (%a, 4)
        # CHECK: %read_1_0_1
        # CHECK-SAME: (%a, 4)

        # CHECK: %b
        # CHECK: %read_0_0_0
        # CHECK-SAME: (%b, 4)
        # CHECK: %read_0_0_1
        # CHECK-SAME: (%b, 4)
        # CHECK: %read_0_1_0
        # CHECK-SAME: (%b, 4)
        # CHECK: %read_0_1_1
        # CHECK-SAME: (%b, 4)

        # CHECK: %mma_0_0_0
        # CHECK-SAME: (%read_0_0_0, %read_0_0_0, %acc)
        # CHECK: %mma_0_0_1
        # CHECK-SAME: (%read_0_0_1, %read_0_0_1, %mma_0_0_0)
        # CHECK: %mma_1_1_0
        # CHECK-SAME: (%read_1_0_0, %read_0_1_0, %acc_1_1_0)
        # CHECK: %mma_1_1_1
        # CHECK-SAME: (%read_1_0_1, %read_0_1_1, %mma_1_1_0)
        # CHECK: %mma_1_0_0
        # CHECK-SAME: (%read_1_0_0, %read_0_0_0, %acc_1_0_0)
        # CHECK: %mma_1_0_1
        # CHECK-SAME: (%read_1_0_1, %read_0_0_1, %mma_1_0_0)
        # CHECK: %mma_0_1_0
        # CHECK-SAME: (%read_0_0_0, %read_0_1_0, %acc_0_1_0)
        # CHECK: %mma_0_1_1
        # CHECK-SAME: (%read_0_0_1, %read_0_1_1, %mma_0_1_0)
        # CHECK: return [mma_0_0_1, mma_1_1_1, mma_1_0_1, mma_0_1_1]

        # CHECK: -----


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
