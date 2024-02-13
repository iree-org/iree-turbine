import logging
import unittest

import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

M = tk.lang.sym.M
K = tk.lang.sym.K


class Test(unittest.TestCase):
    # This test is using the compiler "the hard way" until we have all of the
    # API layering in place.
    def testIotaFx(self):
        @tk.gen.thread(M)
        def iota_kernel(out: tk.lang.OutputBuffer[M]):
            i = tk.lang.program_id(0)
            secret_value = ((i * (33 - i) + 4) % 8) // 2
            out[i] = secret_value

        with tk.gen.TestLaunchContext():
            out = torch.zeros(17, dtype=torch.int32)

    def testSoftmaxFx(self):
        @tk.gen.thread(M)
        def softmax_kernel(
            input: tk.lang.InputBuffer[M, K], output: tk.lang.OutputBuffer[M, K]
        ):
            row_index = tk.lang.program_id(0)
            input_row = input[row_index, :]
            numerator = tkl.exp2(input_row - tkl.max(input_row))
            output_row = numerator / tkl.sum(numerator)
            output[row_index, :] = output_row

        with tk.gen.TestLaunchContext():
            input = torch.randn(128, 64, dtype=torch.float32)
            output = torch.zeros(128, 64, dtype=torch.float32)
            softmax_kernel(input, output)

    def testForLoopFx(self):
        @tk.gen.thread(M)
        def for_loop_kernel(
            input: tk.lang.InputBuffer[M, K], output: tk.lang.OutputBuffer[M, K]
        ):
            row_idx = tkl.program_id(0)
            sum = input[row_idx, 0]
            prefetch = input[row_idx, 1]

            @tkl.for_loop(2, 5, init_args=[sum, prefetch])
            def prefetch_sum(i, sum, prefetch):
                new_sum = sum + prefetch
                new_prefetch = input[row_idx, i]
                return new_sum, new_prefetch

            output[row_idx, 0] = prefetch_sum[0]

        with tk.gen.TestLaunchContext():
            input = torch.randn(128, 64, dtype=torch.float32)
            output = torch.zeros(128, 64, dtype=torch.float32)
            for_loop_kernel(input, output)

    def testGemmFx(self):
        N = tkl.sym.N
        M = tkl.sym.M
        K = tkl.sym.K
        BLOCK_SIZE = tkl.sym.BLOCK_SIZE

        @tk.gen.thread(N // BLOCK_SIZE, M // BLOCK_SIZE)
        def gemm_kernel(
            A: tkl.InputBuffer[N, K],
            B: tkl.InputBuffer[K, M],
            output: tkl.OutputBuffer[N, M],
        ):
            grid_n = tkl.program_id(0)
            grid_m = tkl.program_id(1)

            acc = tkl.constant((BLOCK_SIZE, BLOCK_SIZE), tkl.f32, 0.0)

            @tkl.for_loop(0, K // BLOCK_SIZE, init_args=[acc])
            def body(i, c):
                a = tkl.load(A, (grid_n, i * BLOCK_SIZE), (BLOCK_SIZE, BLOCK_SIZE))
                b = tkl.load(B, (i * BLOCK_SIZE, grid_m), (BLOCK_SIZE, BLOCK_SIZE))
                return (tkl.dot(a, b, c),)

            tkl.store(output, (grid_n, grid_m), body[0])

        with tk.gen.TestLaunchContext({BLOCK_SIZE: 32}):
            A = torch.randn(512, 1024, dtype=torch.float32)
            B = torch.randn(1024, 2048, dtype=torch.float32)
            output = torch.zeros(512, 2048, dtype=torch.float32)
            gemm_kernel(A, B, output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
