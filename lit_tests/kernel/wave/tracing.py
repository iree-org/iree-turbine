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
    # CHECK-SAME: MemoryType[M, N].of(f16)
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
    # CHECK-SAME: MemoryType[M, N].of(f16)
    # CHECK-NEXT: read(memory=a
    # CHECK-NEXT: write(register_=read, memory=a
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
    # CHECK-NEXT: read
    # CHECK-NEXT: read
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


# Input sizes
B = tkl.sym.M
D = tkl.sym.N
S = tkl.sym.K

# Workgroup tile sizes
BLOCK_B = tkl.sym.BLOCK_B
BLOCK_S = tkl.sym.BLOCK_S


@run
def test_attention():
    @tkw.wave_trace_only()
    def attention(
        query: tkl.Memory[B, N, D, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        key: tkl.Memory[B, S, D, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        value: tkl.Memory[B, S, D, ADDRESS_SPACE, tkl.f8e4m3fnuz],
        output: tkl.Memory[B, S, D, ADDRESS_SPACE, tkl.f32],
    ):
        s_reg = tkl.Register[B, N, S, tkl.f32](0.0)
        l_reg = tkl.Register[B, N, tkl.f32](0.0)
        m_reg = tkl.Register[B, N, tkl.f32](-1e3)
        o_reg = tkl.Register[B, N, D, tkl.f32](0.0)
        q = tkw.read(query, elements_per_thread=4)

        @tkw.reduction(S, init_args=[l_reg, m_reg, o_reg])
        def repeat(
            partial_sum: tkl.Register[B, N, tkl.f32],
            partial_max: tkl.Register[B, N, tkl.f32],
            acc: tkl.Register[B, N, D, tkl.f32],
        ) -> tuple[
            tkl.Register[B, N, tkl.f32],
            tkl.Register[B, N, tkl.f32],
            tkl.Register[B, N, D, tkl.f32],
        ]:
            k = tkw.read(key, elements_per_thread=4)
            s = tkw.mma(q, k, s_reg)
            m = tkw.max(s, dim=(S,))
            m = tkw.max(m, partial_max)
            p = tkw.exp2(s - m)
            l = tkw.sum(p, dim=(S,))
            l = tkw.exp2(partial_max - m) * partial_sum + l
            v = tkw.read(value, elements_per_thread=4)
            p = tkw.cast(p, tkl.f8e4m3fnuz)
            acc = tkw.mma(p, v, acc)
            return l, m, acc

        final_sum, _, final_output = repeat
        final_output = final_output / final_sum
        tkw.write(final_output, output, elements_per_thread=4)

    trace = attention()
    print_trace(trace)
    # Root graph:
    # CHECK: %query
    # CHECK-NEXT: %key
    # CHECK-NEXT: %value
    # CHECK-NEXT: %output
    # CHECK-NEXT: %register
    # CHECK-NEXT: %register_1
    # CHECK-NEXT: %register_2
    # CHECK-NEXT: %register_3
    # CHECK-NEXT: %read
    # CHECK-NEXT: %reduction
    # CHECK-NEXT: %getitem
    # CHECK-NEXT: %getitem_1
    # CHECK-NEXT: %getitem_2
    # CHECK-NEXT: %truediv
    # CHECK-NEXT: %write
    # CHECK-NEXT: return None

    # Root graph in custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: register
    # CHECK-NEXT: register
    # CHECK-NEXT: register
    # CHECK-NEXT: register
    # CHECK-NEXT: read
    # CHECK-NEXT: reduction
    # CHECK-NEXT: unknown: getitem
    # CHECK-NEXT: unknown: getitem_1
    # CHECK-NEXT: unknown: getitem_2
    # CHECK-NEXT: unknown: truediv
    # CHECK-NEXT: write
    # CHECK-NEXT: output

    # Subgraph:
    # CHECK: %partial_sum
    # CHECK-NEXT: %partial_max
    # CHECK-NEXT: %acc
    # CHECK-NEXT: %key
    # CHECK-NEXT: %read
    # CHECK-NEXT: %read_1
    # CHECK-NEXT: %register
    # CHECK-NEXT: %mma
    # CHECK-NEXT: %sub
    # CHECK-NEXT: %sub_1
    # CHECK-NEXT: %mul
    # CHECK-NEXT: %add
    # CHECK-NEXT: %value
    # CHECK-NEXT: %read_2
    # CHECK-NEXT: %mma_1
    # CHECK-NEXT: return (add, None, mma_1)

    # Subgraph in custom format:
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: mma
    # CHECK-NEXT: unknown: sub
    # CHECK-NEXT: unknown: sub_1
    # CHECK-NEXT: unknown: mul
    # CHECK-NEXT: unknown: add
    # CHECK-NEXT: placeholder
    # CHECK-NEXT: read
    # CHECK-NEXT: mma
    # CHECK-NEXT: output
