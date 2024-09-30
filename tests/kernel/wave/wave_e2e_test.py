import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.wave.wave_sim import wave_sim
from shark_turbine.kernel.lang.global_symbols import *
import torch
from numpy.testing import assert_allclose, assert_equal
import pytest
import sympy
import os
import torch
import json

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))
require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")
default_test_shapes = [(1, 128), (256, 64), (256, 128), (256, 256), (256, 1024)]

user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_copy(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Min(N, 256)
    ELEMS_PER_THREAD = BLOCK_N / wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    a = torch.randn(shape, dtype=torch.float16)
    b = torch.zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_read"))
def test_transpose_read(shape):
    shape = shape[::-1]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_N = 1
    BLOCK_M = sympy.Min(M, 256)
    ELEMS_PER_THREAD = BLOCK_M / wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={N: BLOCK_N, M: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    a = torch.randn(shape, dtype=torch.float16)
    b = torch.zeros(shape[::-1], dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_write"))
def test_transpose_write(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Min(N, 256)
    ELEMS_PER_THREAD = BLOCK_N / wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i, N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    a = torch.randn(shape, dtype=torch.float16)
    b = torch.zeros(shape[::-1], dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_reduce_sum"))
def test_reduce_sum(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = N
    ELEMS_PER_THREAD = BLOCK_N / wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.sum(res, dim=N)
        tkw.write(res, c, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    torch.manual_seed(1)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    ref = torch.sum((a * b), dim=-1)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b, c)
        assert_allclose(ref, c, atol=0.1)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_reduce_max"))
def test_reduce_max(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = N
    ELEMS_PER_THREAD = BLOCK_N / wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.max(res, dim=N)
        tkw.write(res, c, elements_per_thread=1)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    a = torch.randn(shape, dtype=torch.float16)
    b = torch.randn(shape, dtype=torch.float16)
    c = torch.zeros((shape[0],), dtype=torch.float16)
    ref = torch.max((a * b), dim=-1)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b, c)
        # Assert equal does cast to boolean on torch.Tensor
        # which causes issues, hence we cast to numpy before
        # checking.
        assert_equal(c, ref.values.numpy())


@require_e2e
def test_im2col():
    # TODO: we don't support unaligned access at the moment so all sizes must
    # be aligned to WG/Wave sizes, c * hw * wf == 8 and number of windows == 64.
    n, c, h, w = 1, 2, 9, 9  # Image.
    cf, hf, wf = c, 2, 2  # Filters.
    padding = 0
    stride = 1

    sym = tkl.sym
    ADDRESS_SPACE = sym.ADDRESS_SPACE

    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    # K = HF * WF * C
    # M = SZ_OUT * N
    M = sym.M
    K = sym.K

    # We unroll K dimension according to ELEMS_PER_THREAD value.
    # i.e. for K==8 we will have 2 vector.gather's.
    # Each WG will process 64 windows.
    wave_size = 64
    BLOCK_K = hf * wf * c
    BLOCK_M = 64
    ELEMS_PER_THREAD = 4

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
            W: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
        },
        outputs={M: i, K: j},
    )

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, K: ELEMS_PER_THREAD},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K)]
    # TODO: TilingConstraint doesn't work without actual reduction loop, instead
    # we treat K as WG '1' dimension, but corresponding WG size will be always
    # equal to 1.
    # constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, hf * wf * c)
    a = torch.randn((n, c, h, w), dtype=torch.float16)
    b = torch.zeros(res_shape, dtype=torch.float16)

    im2col = torch.nn.Unfold(kernel_size=(hf, wf), padding=padding, stride=stride)
    expected = im2col(a)[0, :, :].T

    with tk.gen.TestLaunchContext(
        {
            N: n,
            C: c,
            W: w,
            H: h,
            WF: wf,
            HF: hf,
            M: res_shape[0],
            K: res_shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(b, expected)


@require_e2e
def test_im2col_mma():
    # igemm without final col2im
    n, c, h, w = 1, 4, 9, 9  # Image.
    nf, cf, hf, wf = 64, c, 2, 2  # Filters.
    padding = 0  # TODO: only pad=0 is supported for now
    stride = 1

    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

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

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_K = K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            # vector_shapes={NF: 1, M: BLOCK_M, K: ELEMS_PER_THREAD},
        )
    ]

    def func(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[M, NF, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(repeat, out, elements_per_thread=ELEMS_PER_THREAD)

    sim_func = wave_sim(constraints)(func)
    gpu_func = tkw.wave(constraints)(func)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, nf)
    out_ref = torch.zeros(res_shape, dtype=torch.float32)
    sim_func(x, we, out_ref)

    out = torch.zeros_like(out_ref)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    with tk.gen.TestLaunchContext(
        {
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 64,
            BLOCK_N: 64,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        gpu_func(x, we, out)
        assert_allclose(out, out_ref, rtol=1e-05, atol=1e-05)


@require_e2e
def test_igemm_conv():
    n, c, h, w = 1, 4, 5, 5  # Image.
    nf, cf, hf, wf = 16, c, 2, 2  # Filters.
    padding = 0  # TODO: only pad=0 is supported for now
    stride = 1

    torch.manual_seed(1)
    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

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

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
        )
    ]

    @tkw.wave(constraints)
    def conv(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[N, NF, H_OUT, W_OUT, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    out = torch.zeros_like(out_ref)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    with tk.gen.TestLaunchContext(
        {
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 16,
            BLOCK_N: 16,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        conv(x, we, out)
        assert_allclose(out, out_ref, rtol=1e-05, atol=1e-05)
