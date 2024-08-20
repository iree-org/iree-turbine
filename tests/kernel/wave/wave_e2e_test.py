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
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=16)
        tkw.write(res, b, elements_per_thread=16)

    config = {"backend": "rocm", "device": "hip", "target": "gfx942"}

    a = torch.randn(16, 16, dtype=torch.float16)
    b = torch.zeros(16, 16, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: 16,
            N: 16,
            BLOCK_M: 16,
            BLOCK_N: 16,
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b)
        assert_allclose(a, b)
