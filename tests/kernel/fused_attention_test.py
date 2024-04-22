import logging
import unittest

import torch
import shark_turbine.kernel as tk
import shark_turbine.kernel.lang as tkl

BATCH = tkl.sym.BATCH
N_HEADS = tkl.sym.N_HEADS
N_CTX = tkl.sym.N_CTX
D_HEAD = tkl.sym.D_HEAD

BLOCK_N = tkl.sym.BLOCK_N
BLOCK_M = tkl.sym.BLOCK_M


class Test(unittest.TestCase):
    def testFusedAttention(self):
        @tk.gen.thread(N_CTX // BLOCK_M, BATCH * N_HEADS)
        def fused_attention(
            Q: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, tkl.f16],
            K: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, tkl.f16],
            V: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, tkl.f16],
            O: tkl.OutputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, tkl.f16],
        ):
            grid_n = tkl.program_id(0)
            grid_m = tkl.program_id(1)

            batch = grid_m // N_HEADS
            head = grid_m % N_HEADS

            q = tkl.load(Q, (batch, head, grid_n * BLOCK_M, 0), (BLOCK_M, D_HEAD))
            acc_init = tkl.constant((BLOCK_M, D_HEAD), tkl.f32, 0.0)
            max_stat_init = tkl.constant((BLOCK_M,), tkl.f32, -1e9)
            sum_stat_init = tkl.constant((BLOCK_M,), tkl.f32, 0.0)

            @tkl.for_loop(
                0, N_CTX, BLOCK_N, init_args=[max_stat_init, sum_stat_init, acc_init]
            )
            def body(i, old_max, old_sum, old_acc):
                k = tkl.load(K, (batch, head, i, 0), (BLOCK_N, D_HEAD))
                kT = tkl.transpose(k, (1, 0))

                qkT = tkl.constant((BLOCK_M, BLOCK_N), tkl.f32, 0.0)
                qkT = tkl.dot(q, kT, qkT)

                new_max = tkl.max(qkT, axis=1, acc=old_max)
                broadcasted_max = tkl.broadcast_in_dim(
                    new_max, (BLOCK_M, BLOCK_N), (0,)
                )
                partial_softmax = tkl.exp2(qkT - broadcasted_max)
                scale_factor = tkl.exp2(old_max - new_max)
                scaled_old_sum = scale_factor * old_sum
                new_sum = tkl.sum(partial_softmax, axis=1, acc=scaled_old_sum)
                broadcasted_scale_factor = tkl.broadcast_in_dim(
                    scale_factor, (BLOCK_M, D_HEAD), (0,)
                )
                new_acc = old_acc * broadcasted_scale_factor

                v = tkl.load(V, (batch, head, i, 0), (BLOCK_N, D_HEAD))
                qkT16 = tkl.to_dtype(qkT, tkl.f16)
                new_acc = tkl.dot(qkT16, v, new_acc)

                return (new_max, new_sum, new_acc)

            sum_stat = body[1]
            result = body[2]
            one = tkl.constant((BLOCK_M,), tkl.f32, 1.0)
            one_by_sum = one / sum_stat
            result = tkl.broadcast_in_dim(one_by_sum, (BLOCK_M, D_HEAD), (0,)) * result
            tkl.store(O, (batch, head, grid_n * BLOCK_M, 0), result)

        Q = torch.randn(4, 48, 1024, 64)
        K = torch.randn(4, 48, 1024, 64)
        V = torch.randn(4, 48, 1024, 64)
        O = torch.randn(4, 48, 1024, 64)

        with tk.gen.TestLaunchContext(
            {
                BLOCK_N: 128,
                BLOCK_M: 256,
            }
        ):
            fused_attention(Q, K, V, O)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
