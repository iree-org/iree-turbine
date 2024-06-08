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

F8_TYPE = tkl.f8e4m3fnuz

# Product of the Q and K scales for flash attention adjustment
QK_SCALE = 1.1

# Scaling to put QK into a good FP8 range pre-softmax
PRE_SM_SCALE = 1.3

# Apply the final FP8 scaling
OUT_SCALE = 1.5

class Test(unittest.TestCase):
    def testFusedAttention(self):
        @tk.gen.thread(N_CTX // BLOCK_M, BATCH * N_HEADS)
        def fused_attention(
            Q: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, F8_TYPE],
            K: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, F8_TYPE],
            V: tkl.InputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, F8_TYPE],
            O: tkl.OutputBuffer[BATCH, N_HEADS, N_CTX, D_HEAD, F8_TYPE],
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

                # Q and K are linear layers coming from i8-i8 matmul in most
                # cases. It is possible the distribution of these mm is actually
                # limited and not suitable for FP8
                qkT = tkl.dot(q, kT, qkT)

                #### QK FP8 truncation correction
                # We apply the QK scale then compute the softmax values within
                # f32. Given we have `exp` and `sum` we will definitely require
                # higher precision. This attempts to compute the flash-attention
                # update for the softmax affect
                qkT = qkT * tkl.constant((), tkl.f32, QK_SCALE)

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

                #### Softmax FP8 truncation
                # Post computing the softmax correction rescale QK to be within
                # a good FP8 range before applying the next matmul.
                qkT = qkT * tkl.constant((), tkl.f32, PRE_SM_SCALE)
                qkT16 = tkl.to_dtype(qkT, F8_TYPE)

                # We don't need to apply correction for the FP8 biasing yet as each
                # accumulation is applied with the same scaling:
                new_acc = tkl.dot(qkT16, v, new_acc)

                return (new_max, new_sum, new_acc)

            sum_stat = body[1]

            result = body[2]
            one = tkl.constant((BLOCK_M,), tkl.f32, 1.0)
            one_by_sum = one / sum_stat
            result = tkl.broadcast_in_dim(one_by_sum, (BLOCK_M, D_HEAD), (0,)) * result

            #### Result tile FP8 truncation
            # We now rescale the tile result to be within a good FP8 result. This
            # scaling also applies the inverse of the Softmax FP8 truncation scaling.
            result = result * tkl.constant((), tkl.f32, OUT_SCALE)
            result = tkl.to_dtype(result, dtype=F8_TYPE)

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
