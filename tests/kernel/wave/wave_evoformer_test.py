# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import pytest
import torch
from torch.nn import functional as F
import math
import unittest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils import (
    get_default_run_config,
    get_default_arch,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
    device_randn,
    device_zeros,
    device_randint,
)
from iree.turbine.kernel.wave.constraints import MMAType
import os
import json
from torch.testing import assert_close, assert_allclose

_run_e2e = int(os.environ.get("WAVE_RUN_E2E_TESTS", 0))
require_e2e = pytest.mark.skipif(not _run_e2e, reason="e2e tests are disabled")
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(), reason="Default device is not CDNA3"
)
# Whether to dump the generated MLIR module.
test_dump_generated_mlir = int(os.environ.get("WAVE_DUMP_MLIR", 0))
# Whether to use scheduling group barriers (needs LLVM fix).
enable_scheduling_barriers = int(os.environ.get("WAVE_USE_SCHED_BARRIERS", 0))

# Add test shapes for validation and performance testing.
perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
default_test_shapes = {}
# Order of shapes: (B, BN, K2, H, K1, M, N)
default_test_shapes["test_attention"] = [
    (1, 256, 256, 4, 32, 256, 32),
]
default_test_shapes["test_attention"] += [
    perf_test(x) for x in default_test_shapes["test_attention"]
]


user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)

if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes[test_name]


# From: https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/ops/deepspeed4science/test_DS4Sci_EvoformerAttention.py
def attention_reference(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    biases: list[torch.Tensor],
    sm_scale: float,
) -> torch.Tensor:
    q = q_input.transpose(-2, -3)
    k = k_input.transpose(-2, -3)
    v = v_input.transpose(-2, -3)
    k_t = k.transpose(-1, -2)
    a = torch.matmul(q, k_t) * sm_scale

    for b in biases:
        a += b

    a = F.softmax(a, dim=-1)
    a_v = torch.matmul(a, v)
    o = a_v.transpose(-2, -3)

    return o


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_attention"))
@pytest.mark.parametrize("enable_scheduling", [False])
@pytest.mark.parametrize(
    "mfma_variant",
    [
        MMAType.F32_16x16x16_F16,
        MMAType.F32_32x32x8_F16,
    ],
)
def testEvoformerAttentionForward(
    shape: tuple[int],
    enable_scheduling: bool,
    mfma_variant: MMAType,
    request,
):
    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    # Input sizes
    B = tkl.sym.B
    BN = tkl.sym.BN
    M = tkl.sym.M
    H = tkl.sym.H
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_BN = tkl.sym.BLOCK_BN
    BLOCK_H = tkl.sym.BLOCK_H
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
    constraints += [tkw.WorkgroupConstraint(BN, BLOCK_BN, 3)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 4)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

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
            vector_shapes={B: 0, BN: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    m = tkw.IndexMapping.iterator(4)
    # [B, BN, M, H, K1] -> [B, BN, H, M, K1]
    q_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, M: l, K1: m},
        outputs={B: i, BN: j, H: k, M: l, K1: m},
    )
    # [B, BN, K2, H, K1] -> [B, BN, H, K2, K1]
    k_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, K2: l, K1: m},
        outputs={B: i, BN: j, H: k, K2: l, K1: m},
    )
    # [B, BN, N, H, K2] -> [B, BN, H, N, K2]
    v_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, K2: m},
        outputs={B: i, BN: j, H: k, N: l, K2: m},
    )
    # [B, BN, H, N, M] -> [B, BN, M, H, N]
    o_mapping = tkw.IndexMapping(
        num_iterators=5,
        inputs={B: i, BN: j, H: k, N: l, M: m},
        outputs={B: i, BN: j, H: k, N: l, M: m},
    )

    @tkw.wave(constraints)
    def evoformer_fwd(
        q: tkl.Memory[B, BN, M, H, K1, ADDRESS_SPACE, tkl.f16],
        k: tkl.Memory[B, BN, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        v: tkl.Memory[B, BN, N, H, K2, ADDRESS_SPACE, tkl.f16],
        mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        c_reg = tkl.Register[B, BN, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, BN, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, BN, H, M, tkl.f32](-1e6)

        @tkw.reduction(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, BN, H, M, tkl.f32],
            partial_sum: tkl.Register[B, BN, H, M, tkl.f32],
            acc: tkl.Register[B, BN, H, N, M, tkl.f32],
        ) -> (
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, M, tkl.f32],
            tkl.Register[B, BN, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, BN, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(
                q, mapping=q_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            k_reg = tkw.read(
                k, mapping=k_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            inner_acc = tkw.mma(k_reg, q_reg, imm_reg)
            x_j = tkw.permute(inner_acc, target_shape=[B, BN, H, M, K2])
            mask_reg = tkw.read(mask, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_mask_reg = tkw.cast(mask_reg, tkl.f32)
            y_j = x_j + casted_mask_reg
            bias_reg = tkw.read(bias, elements_per_thread=STORE_ELEMS_PER_THREAD)
            casted_bias_reg = tkw.cast(bias_reg, tkl.f32)
            z_j = y_j + casted_bias_reg
            m_j = tkw.max(z_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(z_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            v_reg = tkw.read(
                v, mapping=v_mapping, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        # repeat represents the results of the loop
        res_max, res_sum, res_mm = repeat
        res = res_mm / res_sum
        casted = tkw.cast(res, tkl.f16)
        tkw.write(
            casted, c, mapping=o_mapping, elements_per_thread=STORE_ELEMS_PER_THREAD
        )

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        B: shape[0],
        BN: shape[1],
        K2: shape[2],
        H: shape[3],
        K1: shape[4],
        M: shape[5],
        N: shape[6],
        BLOCK_B: 1,
        BLOCK_BN: 1,
        BLOCK_H: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K2: 32,
        READ_SHARED_DELAY: 1,
        WRITE_SHARED_DELAY: 1,
        READ_GLOBAL_DELAY: 2,
        WRITE_GLOBAL_DELAY: 2,
        MMA_DELAY: 1,
        VALU_DELAY: 1,
        SHUFFLE_DELAY: 1,
        SHARED_MEMORY_UNITS: 4,
        GLOBAL_MEMORY_UNITS: 4,
        MMA_UNITS: 4,
        VALU_UNITS: 2,
        SHUFFLE_UNITS: 2,
    }
    config = get_default_run_config()
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=enable_scheduling,
        use_scheduling_barriers=enable_scheduling_barriers,
    ):
        torch.manual_seed(0)
        batch, n, kv_seq_len, heads, head_dim, q_seq_len, v_dim = shape
        # q: tkl.Memory[B, BN, M, H, K1, ADDRESS_SPACE, tkl.f16],
        q = device_randn(batch, n, q_seq_len, heads, head_dim, dtype=torch.float16)
        # k: tkl.Memory[B, BN, K2, H, K1, ADDRESS_SPACE, tkl.f16],
        k = device_randn(batch, n, kv_seq_len, heads, head_dim, dtype=torch.float16)
        # v: tkl.Memory[B, BN, K2, H, N, ADDRESS_SPACE, tkl.f16],
        v = device_randn(batch, n, kv_seq_len, heads, v_dim, dtype=torch.float16)
        # mask: tkl.Memory[B, BN, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        mask = device_randint(0, 2, (batch, n, kv_seq_len), dtype=torch.float16)
        mask_bias = 1e9 * (mask - 1)
        # bias: tkl.Memory[B, H, M, K2, GLOBAL_ADDRESS_SPACE, tkl.f16],
        bias = device_randn(batch, heads, q_seq_len, kv_seq_len, dtype=torch.float16)
        # output: tkl.Memory[B, BN, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
        output = device_zeros(batch, n, q_seq_len, heads, v_dim, dtype=torch.float16)
        log2e = 1.44269504089
        dk_sqrt = math.sqrt(1.0 / shape[4])
        # TODO: Add scaling of QK as part of kernel.
        mb = evoformer_fwd(
            q * dk_sqrt * log2e,
            k,
            v.permute([0, 1, 4, 3, 2]),
            mask_bias,
            bias * log2e,
            output,
        )

        mask_bias = mask_bias.view([batch, n, 1, 1, kv_seq_len])
        bias = bias.view([batch, 1, heads, q_seq_len, kv_seq_len])
        torch_ref = attention_reference(q, k, v, [mask_bias, bias], dk_sqrt)

        if test_dump_generated_mlir:
            filename = f"wave_evoformer_{'x'.join(map(str, shape))}.mlir"
            with open(filename, "w") as f:
                f.write(mb.module_op.get_asm())

        eps = 1e-2 if output.dtype == torch.float16 else 5e-2
        print(f"Max diff: {torch.max(torch.abs(torch_ref - output)).item()}")
        assert (
            torch.max(torch.abs(torch_ref - output)).item() < eps
        ), f"out eps: {torch.max(torch.abs(torch_ref - output))}"
