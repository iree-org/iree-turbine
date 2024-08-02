import logging
import pytest
import torch
import unittest
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl
import shark_turbine.kernel.wave as tkw


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
            tkw.HardwareConstraint(threads_per_wave=64, waves_per_block=(1, 1, 1))
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

        hyperparams = {
            ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 1,
            BLOCK_M: 32,
            BLOCK_N: 32,
            BLOCK_K: 32,
            M: 64,
            N: 128,
            K: 256,
        }
        with pytest.raises(
            NotImplementedError, match="Currently only stub implementation"
        ):
            with tk.gen.TestLaunchContext(hyperparams):
                a = torch.randn(64, 256, dtype=torch.float16)
                b = torch.randn(128, 256, dtype=torch.float16)
                c = torch.zeros(64, 128, dtype=torch.float32)
                gemm(a, b, c)

                # TODO: Note this is currently not triggered as the stub exception
                # is raised first. Remove this note when this successfully runs
                # through codegen.
                assert gemm.grid_type.symbolic_shape == (
                    M // BLOCK_M,
                    N // BLOCK_N,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
