import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
import torch
from numpy.testing import assert_allclose
import pytest
import os

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))

require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")


@require_e2e
def test_copy():
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: ELEMS_PER_THREAD},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, 64)]
    constraints += [tkw.WaveConstraint(N, 1)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    shape = (16, 16)
    a = torch.randn(shape, dtype=torch.float16)
    b = torch.zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_M: 16,
            BLOCK_N: 16,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(a, b)
