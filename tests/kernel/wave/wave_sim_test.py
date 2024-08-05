import pytest
import torch
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.wave_sim import wave_sim
from numpy.testing import assert_allclose


def test_eltwise():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def eltwise(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.write(a_reg + b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    a = torch.randn(128, 256, dtype=torch.float32)
    b = torch.randn(128, 256, dtype=torch.float32)
    c = torch.zeros(128, 256, dtype=torch.float32)
    eltwise(a, b, c)
    assert_allclose(c, a + b)


def test_broadcast_1():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def eltwise(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.write(a_reg + b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    a = torch.randn(128, 256, dtype=torch.float32)
    b = torch.randn(256, dtype=torch.float32)
    c = torch.zeros(128, 256, dtype=torch.float32)
    eltwise(a, b, c)
    assert_allclose(c, a + b)


def test_broadcast_2():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def eltwise(
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.write(b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    b = torch.randn(256, dtype=torch.float32)
    c = torch.zeros(128, 256, dtype=torch.float32)
    eltwise(b, c)
    assert_allclose(c, b + torch.zeros(128, 256, dtype=torch.float32))


def test_broadcast_3():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def eltwise(
        b: tkl.Memory[N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)[0]
        tkw.write(b_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    b = torch.randn(256, dtype=torch.float32)
    c = torch.zeros(128, 256, dtype=torch.float32)
    eltwise(b, c)
    assert_allclose(c, b[0] + torch.zeros(128, 256, dtype=torch.float32))


def test_gemm():
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
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    @wave_sim(constraints)
    def gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        # This microkernel encodes the fact that if the reduction
        # dimension were tiled, then we would need to materialize a loop.
        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
            # a_reg: tkw.Register[M, K, tkl.f16]
            a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # b_reg: tkw.Register[N, K, tkl.f16]
            b_reg = tkw.read(b, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            # acc: tkw.Register[M, N, tkl.f32]
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        # repeat represents the results of the loop
        tkw.write(repeat, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    a = torch.randn(64, 256, dtype=torch.float16)
    b = torch.randn(128, 256, dtype=torch.float16)
    c = torch.zeros(64, 128, dtype=torch.float32)
    gemm(a, b, c)
    assert_allclose(c, a @ b.T)


def test_transpose_1():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @wave_sim(constraints)
    def transpose(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, mapping=mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.write(a_reg, c, elements_per_thread=STORE_ELEMS_PER_THREAD)

    a = torch.randn(128, 256, dtype=torch.float32)
    c = torch.zeros(256, 128, dtype=torch.float32)
    transpose(a, c)
    assert_allclose(c, a.T)


def test_transpose_2():
    # Input sizes
    M = tkl.sym.M
    N = tkl.sym.N
    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @wave_sim(constraints)
    def transpose(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(a, elements_per_thread=LOAD_ELEMS_PER_THREAD)
        tkw.write(
            a_reg,
            c,
            mapping=mapping,
            elements_per_thread=STORE_ELEMS_PER_THREAD,
        )

    a = torch.randn(128, 256, dtype=torch.float32)
    c = torch.zeros(256, 128, dtype=torch.float32)
    transpose(a, c)
    assert_allclose(c, a.T)


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("c", [1, 3, 10])
@pytest.mark.parametrize("nf", [1, 2, 8])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_igemm_conv(n, c, nf, stride):
    h, w = 5, 5  # Image.
    cf, hf, wf = c, 2, 2  # Filters.
    x = torch.randn(n, c, h, w, dtype=torch.float32)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float32)

    padding = 0  # TODO: only pad=0 is supported for now
    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach()

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
            W: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j // (HF * WF), HF: j % WF, WF: (j % (HF * WF)) // WF},
        outputs={NF: i, K: j},
    )
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def conv(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[N, NF, H_OUT, W_OUT, ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    out = torch.zeros_like(out_ref)
    conv(x, we, out)
    assert_allclose(out, out_ref, rtol=1e-05, atol=1e-05)
