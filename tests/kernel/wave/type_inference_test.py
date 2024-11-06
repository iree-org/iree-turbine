# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import logging
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.ops.wave_ops import get_custom


class TypeInferenceTest(unittest.TestCase):
    def testAttentionInference(self):
        shape = (8, 128, 128, 64, 256)
        # Input sizes
        B = tkl.sym.B
        M = tkl.sym.M
        N = tkl.sym.N
        K1 = tkl.sym.K1
        K2 = tkl.sym.K2
        # Workgroup tile sizes
        BLOCK_B = tkl.sym.BLOCK_B
        BLOCK_M = tkl.sym.BLOCK_M
        BLOCK_N = tkl.sym.BLOCK_N
        BLOCK_K2 = tkl.sym.BLOCK_K2
        # Address space (for GPU, shared(1) or global(0))
        ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
        # Other hyperparameters
        LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
        STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

        # Expose user-constraints
        constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
        constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
        constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
        constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
        constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
        constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

        mfma_variant = MMAType.F32_16x16x16_F16
        if mfma_variant == MMAType.F32_16x16x16_F16:
            Mvec = 16
            Nvec = 16
        if mfma_variant == MMAType.F32_32x32x8_F16:
            Mvec = 32
            Nvec = 32

        constraints += [
            tkw.HardwareConstraint(
                threads_per_wave=64,
                waves_per_block=(2, 2, 1),
                mma_type=mfma_variant,
                vector_shapes={B: 0, M: Mvec, N: Nvec},
            )
        ]

        i = tkw.IndexMapping.iterator(0)
        j = tkw.IndexMapping.iterator(1)
        k = tkw.IndexMapping.iterator(2)
        mapping = tkw.IndexMapping(
            num_iterators=3, inputs={B: i, N: j, M: k}, outputs={B: i, M: k, N: j}
        )

        @tkw.wave_trace_only(constraints)
        def base_attention(
            q: tkl.Memory[B, M, K1, ADDRESS_SPACE, tkl.f16],
            k: tkl.Memory[B, K2, K1, ADDRESS_SPACE, tkl.f16],
            v: tkl.Memory[B, N, K2, ADDRESS_SPACE, tkl.f16],
            c: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
        ):
            c_reg = tkl.Register[B, N, M, tkl.f32](0.0)
            init_sum = tkl.Register[B, M, tkl.f32](0.0)
            init_max = tkl.Register[B, M, tkl.f32](-1e6)

            # This microkernel encodes the fact that if the reduction
            # dimension were tiled, then we would need to materialize a loop.
            @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
            def repeat(
                partial_max: tkl.Register[B, M, tkl.f32],
                partial_sum: tkl.Register[B, M, tkl.f32],
                acc: tkl.Register[B, N, M, tkl.f32],
            ) -> (
                tkl.Register[B, M, tkl.f32],
                tkl.Register[B, M, tkl.f32],
                tkl.Register[B, N, M, tkl.f32],
            ):
                imm_reg = tkl.Register[B, K2, M, tkl.f32](0.0)
                q_reg = tkw.read(q, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                # b_reg: tkw.Register[B, N, K, tkl.f16]
                k_reg = tkw.read(k, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                # acc: tkw.Register[B, N, M, tkl.f32]
                inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
                x_j = tkw.permute(inner_acc, target_shape=[B, M, K2])
                m_j = tkw.max(x_j, partial_max, dim=K2)
                e_delta_max = tkw.exp2(partial_max - m_j)
                e_delta = tkw.exp2(x_j - m_j)
                e_init = partial_sum * e_delta_max
                d_j = tkw.sum(e_delta, e_init, dim=K2)
                imm_f16 = tkw.cast(e_delta, tkl.f16)
                v_reg = tkw.read(v, elements_per_thread=LOAD_ELEMS_PER_THREAD)
                new_acc = acc * e_delta_max
                acc = tkw.mma(v_reg, imm_f16, new_acc)
                return m_j, d_j, acc

            # repeat represents the results of the loop
            res_max, res_sum, res_mm = repeat
            res = res_mm / res_sum
            tkw.write(
                res, c, mapping=mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
            )

        hyperparams = {
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
            BLOCK_B: 1,
            BLOCK_M: 64,
            BLOCK_N: 64,
            BLOCK_K2: 32,
            B: shape[0],
            M: shape[1],
            N: shape[2],
            K1: shape[3],
            K2: shape[4],
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
        }
        with tk.gen.TestLaunchContext(
            hyperparams,
            canonicalize=True,
            run=False,
            run_bench=False,
            schedule=False,
            use_scheduling_barriers=False,
        ):
            trace: CapturedTrace = base_attention()
            IndexingContext.current().finalize()
            infer_types(trace)
            expected_type = {
                "partial_sum": "Register[B, M].of(f32)",
                "partial_max": "Register[B, M].of(f32)",
                "acc": "Register[B, N, M].of(f32)",
                "q": "Memory[B, M, K1].of(f16)",
                "read": "Register[B, M, K1].of(f16)",
                "k": "Memory[B, K2, K1].of(f16)",
                "read_1": "Register[B, K2, K1].of(f16)",
                "mma": "Register[B, K2, M].of(f32)",
                "permute": "Register[B, M, K2].of(f32)",
                "max_1": "Register[B, M].of(f32)",
                "sub": "Register[B, M].of(f32)",
                "exp2": "Register[B, M].of(f32)",
                "sub_1": "Register[B, M, K2].of(f32)",
                "exp2_1": "Register[B, M, K2].of(f32)",
                "mul": "Register[B, M].of(f32)",
                "sum_1": "Register[B, M].of(f32)",
                "cast": "Register[B, M, K2].of(f16)",
                "v": "Memory[B, N, K2].of(f16)",
                "read_2": "Register[B, N, K2].of(f16)",
                "mul_1": "Register[B, N, M].of(f32)",
                "mma_1": "Register[B, N, M].of(f32)",
                "c": "Memory[B, M, N].of(f32)",
                "register_1": "Register[B, M].of(f32)",
                "register_2": "Register[B, M].of(f32)",
                "reduction": "[Register[B, M].of(f32), Register[B, M].of(f32), Register[B, N, M].of(f32)]",
                "getitem": "Register[B, M].of(f32)",
                "getitem_1": "Register[B, M].of(f32)",
                "getitem_2": "Register[B, N, M].of(f32)",
                "truediv": "Register[B, N, M].of(f32)",
                "write": "Memory[B, N, M].of(f32)",
            }
            for subgraph in trace.region_graph.subgraphs.values():
                for node in subgraph.nodes:
                    custom = get_custom(node)
                    if custom.fx_node.name in expected_type:
                        assert str(custom.type) == expected_type[custom.fx_node.name]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
