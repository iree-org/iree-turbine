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
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]

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

    mapping = tkw.IndexMapping(lambda i, j: (j, i))

    @wave_sim(constraints)
    def transpose(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f32],
    ):
        a_reg = tkw.read(
            a, mapping=mapping, shape=(N, M), elements_per_thread=LOAD_ELEMS_PER_THREAD
        )
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

    mapping = tkw.IndexMapping(lambda i, j: (j, i))

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


def test_igemm_conv():
    n, c, h, w = 2, 3, 4, 4  # Image.
    nf, cf, hf, wf = 2, c, 2, 2  # Filters.
    x = torch.randn(n, c, h, w, dtype=torch.float32)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float32)

    stride = 2
    padding = 0  # TODO: only pad=0 is supported for now
    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach()
    print("src")
    print(x)
    print("weight")
    print(we)
    print("res")
    print(out_ref)

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF
    KB = sym.KB
    H_OUT = (H + 2 * padding - HF) / stride + 1
    W_OUT = (W + 2 * padding - WF) / stride + 1
    SZ_OUT = H_OUT * W_OUT
    # SZ_OUT_N = SZ_OUT *

    x_mapping = tkw.IndexMapping(
        lambda i, j: (
            i // SZ_OUT,
            j // (HF * WF),
            (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
            (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
        )
    )
    w_mapping = tkw.IndexMapping(
        lambda i, j: (i % NF, j // (HF * WF), j % WF, (j % (HF * WF)) // WF)
    )
    out_mapping = tkw.IndexMapping(
        lambda i, j: (i // SZ_OUT, j, (i % SZ_OUT) % W_OUT, (i % SZ_OUT) // W_OUT)
    )

    K = HF * WF * C
    M = SZ_OUT * N

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
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 2)]

    constraints += [
        tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
    ]

    @wave_sim(constraints)
    def conv(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[N, NF, H_OUT, W_OUT, ADDRESS_SPACE, tkl.f32],
    ):
        print("-=-=-=-=-=-=-")
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                shape=(M, K),
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
            )
            print(a_reg)
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                shape=(NF, K),
                elements_per_thread=LOAD_ELEMS_PER_THREAD,
            )
            print(b_reg)
            acc = tkw.mma(a_reg, b_reg, acc)
            print(acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    out = torch.zeros_like(out_ref)
    conv(x, we, out)
    print(out)
    assert_allclose(out, out_ref, rtol=1e-05)
