# RUN: python %s | FileCheck %s

from typing import Callable
from shark_turbine.kernel._support.tracing import CapturedTrace
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.ops.wave_ops import get_custom, Read, Write

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


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


@run
def test_trace_empty():
    @tkw.wave_trace_only()
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        pass

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-SAME: _type=Memory[M, N].of(f16)
    # CHECK-NEXT: output


@run
def test_trace_empty_then_add_nodes():
    """
    This tests the modification of a graph after the trace has been created.
    """

    @tkw.wave_trace_only()
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        pass

    trace = test()

    graph = trace.get_root_graph()
    a = list(graph.nodes)[0]
    # Insert at the end of the graph
    with graph.inserting_before(list(graph.nodes)[-1]):
        read = Read(a).add_to_graph(graph)
        write = Write(read, a, 4).add_to_graph(graph)

    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-SAME: _type=Memory[M, N].of(f16)
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: write(register_=read, memory=a
    # CHECK-NEXT: output


@run
def test_trace_py_arithmetic():
    @tkw.wave_trace_only()
    def test(A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        a = tkw.read(A)
        res = a + a - a
        res = -res
        tkw.write(res, A, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-SAME: (%a, None, None)
    # CHECK-NEXT: %add
    # CHECK-SAME: (%read, %read)
    # CHECK-NEXT: %sub
    # CHECK-SAME: (%add, %read)
    # CHECK-NEXT: %neg
    # CHECK-SAME: (%sub,)
    # CHECK-NEXT: %write
    # CHECK-SAME: (%neg, %a, 4, None)
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: add(lhs=read, rhs=read)
    # CHECK-NEXT: sub(lhs=add, rhs=read)
    # CHECK-NEXT: neg(arg=sub)
    # CHECK-NEXT: write(register_=neg, memory=a, elements_per_thread=4)
    # CHECK-NEXT: output


@run
def test_trace_read():
    @tkw.wave_trace_only()
    def test(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        tkw.read(a)

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: output


@run
def test_trace_register():
    @tkw.wave_trace_only()
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        tkw.register([M, N], tkl.f16, 0.0)

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %register
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: register
    # CHECK-SAME: shape=[M, N], dtype=f16
    # CHECK-NEXT: output


@run
def test_trace_write():
    @tkw.wave_trace_only()
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        val = tkw.register([M, N], tkl.f16, 0.0)
        tkw.write(val, a, elements_per_thread=4)

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %register
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: register
    # CHECK-NEXT: write
    # CHECK-NEXT: output


@run
def test_trace_mma():
    @tkw.wave_trace_only()
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        reg_0 = tkw.read(a)
        reg_1 = tkw.read(a)
        acc = tkl.Register[M, N, tkl.f32](0.0)
        mma = tkw.mma(reg_0, reg_1, acc)

    trace = test()
    print_trace(trace)
    # CHECK: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read
    # CHECK-NEXT: %register
    # CHECK-NEXT: %mma
    # CHECK-NEXT: return None

    # Custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: register
    # CHECK-NEXT: mma
    # CHECK-NEXT: output


@run
def test_trace_gemm():
    @tkw.wave_trace_only()
    def gemm(
        A: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        B: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        C: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        c = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c])
        def repeat(acc) -> tkl.Register[M, N, tkl.f32]:
            a = tkw.read(A)
            b = tkw.read(B)
            acc = tkl.Register[M, N, tkl.f32](0.0)
            mma = tkw.mma(a, b, acc)
            return acc

        tkw.write(repeat, C, elements_per_thread=4)

    trace = gemm()
    print_trace(trace)
    # Root graph:
    # CHECK: %a
    # CHECK-NEXT: %b
    # CHECK-NEXT: %c
    # CHECK-NEXT: %register
    # CHECK-NEXT: %reduction
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # Root graph in custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: register
    # CHECK-NEXT: reduction
    # CHECK-NEXT: write
    # CHECK-NEXT: output

    # Subgraph:
    # CHECK: %acc
    # CHECK-NEXT: %a
    # CHECK-NEXT: %read
    # CHECK-NEXT: %b
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %register
    # CHECK-NEXT: %mma
    # CHECK-NEXT: return register

    # Subgraph in custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read
    # CHECK-NEXT: register
    # CHECK-NEXT: mma
    # CHECK-NEXT: output
