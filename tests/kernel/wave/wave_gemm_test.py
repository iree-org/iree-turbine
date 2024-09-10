import logging
import pytest
import torch
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw
from shark_turbine.kernel.lang.global_symbols import *
from shark_turbine.kernel.wave.iree_utils import generate_iree_ref
import os
from numpy.testing import assert_allclose

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))

require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")


@require_e2e
class Test(unittest.TestCase):
    def testGemm(self):

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
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

        constraints += [
            tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(2, 2, 1))
        ]

        # Wave-level micro-kernel.
        # Since warps are not directly addressable, there is no
        # explicit notion of a warp id (like a workgroup or thread id).
        # This kernel uses the input sizes M, N, K throughout, as the tiling
        # and data movement strategy is determined during the compilation process.
        # These can be influenced by introducing constraints.
        @tkw.wave(constraints)
        def gemm(
            a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
            b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
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

        hyperparams = {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K: 32,
            M: 2048,
            N: 10240,
            K: 1280,
        }
        config = {"backend": "rocm", "device": "hip", "target": "gfx942"}
        with tk.gen.TestLaunchContext(
            hyperparams, canonicalize=True, run=True, run_config=config
        ):
            a = torch.randn(2048, 1280, dtype=torch.float16)
            b = torch.randn(10240, 1280, dtype=torch.float16)
            c = torch.zeros(2048, 10240, dtype=torch.float32)
            gemm(a, b, c)
            iree_ref = torch.zeros(2048, 10240, dtype=torch.float32)
            generate_iree_ref("mmt", [a, b], [iree_ref], config)
            assert_allclose(c.numpy(), iree_ref.numpy())


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
