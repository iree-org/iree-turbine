# RUN: python %s | FileCheck %s

import copy
import logging
from typing import Sequence

import torch
import torch.nn.functional as F

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph
from iree.turbine.kernel.wave.analysis.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.compiler.ir import Context, Location, Module
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.wave import LaunchableWave
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
    run_test,
)
from iree.turbine.kernel.wave.utils.print_utils import print_trace, try_apply_pass
from iree.turbine.kernel.wave.utils.compile_utils import (
    set_default_compile_config,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)

DOC = """
# MoE TCTK
This is a TCTK version of the MoE kernel. It is looking at the problem from first principles.

Let's have fun with this one

# Torch Naive MoE
# https://github.com/sgl-project/sglang/blob/7e3bb5270524d38bd98b93a22441fa693c3fa64c/test/srt/test_fused_moe.py#L47
# Triton MoE
# https://github.com/sgl-project/sglang/blob/ffa1b3e318c9d1342a5e430eb04df609e22a3775/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py#L1159
    def torch_naive_moe(self, a, w1, w2, score, topk):
        # where
        # a: Tensor[B, D1],
        # W1: Tensor[E, 2 * N, D1],
        # W2: Tensor[E, D2, N],
        # score: Tensor[B, E], topk: int
        # out: Tensor[B, TOPK, D2]
        B, D = a.shape
        a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
        out = torch.zeros(B * topk,
                          w2.shape[1],
                          dtype=a.dtype,
                          device=a.device)
        score = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight, topk_ids = torch.topk(score, topk)
        topk_weight = topk_weight.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = silu_and_mul(
                    a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
        return (out.view(B, -1, w2.shape[1]) *
                topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)

## TCTK notation

# score: Tensor[B, E],
topk_score[B, TOPK], topk_idx[B, TOPK] = \
    torch.topk(score[B, E])

# Note: softmax normalization only needed on the TOPK subset
m_s[B] max= score[B, :]
score[B, :] -= m_s[B] # for softmax stability
sum_exp[B] += torch.exp(score[B, :]) # for actual softmax normalization
topk_score[B, TOPK] = exp(topk_score[B, TOPK] - m_s[B]) / sum_exp[B]

TMP[TOPK, B, 2 * N] = A[B, D1] @ W1[subset(E by topk_idx[B, TOPK]), 2 * N, D1].transpose(0, 1)
TMP_2[TOPK, B, N] = SILU(TMP[TOPK, B, :N]) * TMP[TOPK, B, N:]
TMP_3[TOPK, B, D2] = TMP_2[TOPK, B, N] @ W2[subset(E by topk_idx[B, TOPK]), D2, N].transpose(0, 1)

RESULT[B, TOPK, D2] = TMP_3[TOPK, B, D2] * topk_score[B, TOPK]

Instead we'll use TOPK, B, D2 for the output tensor because TK does not like
1-broadcast dimensions in another form.
RESULT[B, TOPK, D2] = TMP_3[TOPK, B, D2] * topk_score[B, TOPK]
"""

# Static shapes
vB = 32
vN = 32
vD1 = 32
vD2 = 32
vTOPK = 5
vE = 128

# Symbols
B, N, D1, D2, TOPK = (
    tkl.sym.B,
    tkl.sym.N,
    tkl.sym.D1,
    tkl.sym.D2,
    tkl.sym.TOPK,
)

BLOCK_B, BLOCK_N, BLOCK_D1, BLOCK_D2, BLOCK_TOPK = (
    tkl.sym.BLOCK_B,
    tkl.sym.BLOCK_N,
    tkl.sym.BLOCK_D1,
    tkl.sym.BLOCK_D2,
    tkl.sym.BLOCK_TOPK,
)

LOAD_ELEMS_PER_THREAD, STORE_ELEMS_PER_THREAD = (
    tkl.sym.LOAD_TOKS_PER_THREAD,
    tkl.sym.STORE_ELEMS_PER_THREAD,
)

ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def lit_harness(build_constraints_fun, kernel_fun, *args, **kwargs):
    constraints = build_constraints_fun(*args, **kwargs)
    with tk.gen.TestLaunchContext(
        kwargs["static_symbols"] if "static_symbols" in kwargs else {}
    ):
        lw = LaunchableWave(constraints, "kernel_fun", kernel_fun)

        trace: CapturedTrace = lw._trace()
        idxc: IndexingContext = IndexingContext.current()
        graph_passes = lw.build_initial_pass_pipeline(trace, print_ir_before=["all"])
        for p in graph_passes:
            try_apply_pass(p, trace, print_ir_before=["all"])

        lw.infer_grid_shape(idxc)

        assert False, "TODO: transition to set_default_compile_config post rebase"
        compile_config = get_default_compile_config()
        with Context() as context:
            mb, trace, exe, kernel_sig, entrypoint_name = lw.compile_to_mlir(
                trace, context, **kwargs, compile_config=compile_config
            )
            print(mb.module_op)


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [
        tkw.WorkgroupConstraint(B, BLOCK_B, 0),
        tkw.WaveConstraint(B, BLOCK_B),
    ]
    constraints += [
        tkw.WorkgroupConstraint(D2, BLOCK_D2, 1),
        tkw.WaveConstraint(D2, BLOCK_D2),
    ]
    constraints += [tkw.WorkgroupConstraint(TOPK, BLOCK_TOPK, 2)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            # One must always specify mma_type or vector_shapes.
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes=kwargs["vector_shapes"] if "vector_shapes" in kwargs else {},
        )
    ]
    return constraints


def config(mma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16):
    return {
        "static_symbols": {
            B: vB,
            N: vN,
            D1: vD1,
            D2: vD2,
            TOPK: vTOPK,
            BLOCK_B: 16,
            BLOCK_N: 16,
            BLOCK_D1: 16,
            BLOCK_D2: 16,
            BLOCK_TOPK: 1,
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mma_variant),
        },
        "vector_shapes": {
            TOPK: 0,
            # D1: 16,  # TODO: connected to MFMA op type
            # N: 16,
            # B: 16,
        },
        "canonicalize": {True},
    }


i, j, k = [tkw.IndexMapping.iterator(i) for i in range(3)]
d0 = tkw.IndexMapping.dynamic_val(0)

offset_mapping_w1_lo = tkw.IndexMapping(
    num_iterators=3,
    inputs={TOPK: d0, N: j, D1: k},
    outputs={TOPK: i, N: j, D1: k},
    dynamic_val_mappings=({B: i, TOPK: j}),
)

offset_mapping_w1_hi = tkw.IndexMapping(
    num_iterators=3,
    inputs={TOPK: d0, N: j, D1: k + 777},
    outputs={TOPK: i, N: j, D1: k},
    dynamic_val_mappings=({B: i, TOPK: j}),
)

offset_mapping_tmp_shift = tkw.IndexMapping(
    num_iterators=3,
    inputs={TOPK: i, B: j, N: k + 777},  # TODO: plug in the actual shift value
    outputs={TOPK: i, B: j, N: k},
)

offset_mapping_w2 = tkw.IndexMapping(
    num_iterators=3,
    inputs={TOPK: d0, D2: j, N: k},
    outputs={TOPK: i, D2: j, N: k},
    dynamic_val_mappings=({B: i, TOPK: j}),
)


def fused_moe_kernel(
    TMP_2: tkl.Memory[TOPK, B, N, ADDRESS_SPACE, tkl.f16],
    # Note: W2 really has torch.Tensor.shape [E, D2, N] but we want to index it
    # with indices [B, TOPK, D2, N].
    # We don't want to introduce index E because we'd get the cartesian product
    # [E, B, TOPK], which is not what we want.
    # So we just use TOPK to index into W2 (alternatively we could use B).
    W2: tkl.Memory[TOPK, D2, N, ADDRESS_SPACE, tkl.f16],
    TOPK_INDICES: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.i64],
    TOPK_WEIGHTS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.f32],
    RESULT: tkl.Memory[TOPK, B, D2, ADDRESS_SPACE, tkl.f32],
):
    res_reg = tkl.Register[TOPK, B, D2, tkl.f32](0.0)

    @tkw.reduction(N, init_args=[res_reg])
    def repeat(
        acc: tkl.Register[TOPK, B, D2, tkl.f32]
    ) -> tkl.Register[TOPK, B, D2, tkl.f32]:
        ###
        # TMP_3[TOPK, B, D2] = TMP_2[TOPK, B, N:]
        #   @ W2[subset(E by topk_idx[B, TOPK]), D2, N].transpose(0, 1)
        ###
        expert_id = tkw.read(
            TOPK_INDICES,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )  # : [B, TOPK]
        tmp_2_reg = tkw.read(
            TMP_2,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )  # : [TOPK, B, N]
        w2_reg = tkw.read(
            W2,
            mapping=offset_mapping_w2,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )  # : [TOPK, D2, N] but indexed as [E=(B, TOPK), D2, N] and E expert_id
        acc = tkw.mma(tmp_2_reg, w2_reg, acc)
        return acc

    ###
    # RESULT[TOPK, B, D2] = TMP_3[TOPK, B, D2] * topk_score[B, TOPK]
    ###
    topk_weight = tkw.read(
        TOPK_WEIGHTS, elements_per_thread=LOAD_ELEMS_PER_THREAD
    )  # : [B, TOPK]
    # topk_weight = tkw.permute(topk_weight, target_shape=[TOPK, B])
    res = repeat * topk_weight  # : [TOPK, B, D2]
    tkw.write(
        res, RESULT, elements_per_thread=STORE_ELEMS_PER_THREAD
    )  # : [TOPK, B, D2]


def get_fused_moe_kernel(constraints):
    @tkw.wave(constraints)
    def fused_moe_kernel_executable(
        TMP_2: tkl.Memory[TOPK, B, N, ADDRESS_SPACE, tkl.f16],
        # Note: W2 really has torch.Tensor.shape [E, D2, N] but we want to index it
        # with indices [B, TOPK, D2, N].
        # We don't want to introduce index E because we'd get the cartesian product
        # [E, B, TOPK], which is not what we want.
        # So we just use TOPK to index into W2 (alternatively we could use B).
        W2: tkl.Memory[TOPK, D2, N, ADDRESS_SPACE, tkl.f16],
        TOPK_INDICES: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.i64],
        TOPK_WEIGHTS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.f32],
        RESULT: tkl.Memory[TOPK, B, D2, ADDRESS_SPACE, tkl.f32],
    ):
        return fused_moe_kernel(TMP_2, W2, TOPK_INDICES, TOPK_WEIGHTS, RESULT)

    return fused_moe_kernel_executable


def silu_and_mul(x: torch.Tensor):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_naive_moe_step_1(a, w1, w2, score, topk):
    # where
    # a: Tensor[B, D1],
    # W1: Tensor[E, 2 * N, D1],
    # W2: Tensor[E, D2, N],
    # score: Tensor[B, E], topk: int
    # out: Tensor[B, TOPK, D2]
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1))
    return topk_weight, topk_ids, out


def torch_naive_moe_step_2(a, w1, w2, score, topk):
    topk_weight, topk_ids, out = torch_naive_moe_step_1(a, w1, w2, score, topk)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = out[mask] @ w2[i].transpose(0, 1)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def torch_naive_moe(a, w1, w2, score, topk):
    # where
    # a: Tensor[B, D1],
    # W1: Tensor[E, 2 * N, D1],
    # W2: Tensor[E, D2, N],
    # score: Tensor[B, E], topk: int
    # out: Tensor[B, TOPK, D2]
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


if __name__ == "__main__":

    @run_test
    def static_correct_1():
        cfg = copy.deepcopy(config())
        lit_harness(build_block_constraints, fused_moe_kernel, **cfg)

        a = torch.randn(vB, vD1)
        w1 = torch.randn(vE, 2 * vN, vD1)
        w2 = torch.randn(vE, vD2, vN)
        score = torch.randn(vB, vE)
        topk = vTOPK
        reference = torch_naive_moe(a, w1, w2, score, topk)

        cfg = config()
        assert False, "TODO: transition to set_default_run_config post rebase"
        executable_kernel = get_fused_moe_kernel(build_block_constraints(cfg))
        with tk.gen.TestLaunchContext(
            cfg["static_symbols"] if "static_symbols" in cfg else {},
            canonicalize=True,
            run=True,
            run_config=get_default_run_config(),
        ):
            topk_weight, topk_ids, tmp_2 = torch_naive_moe_step_1(
                a, w1, w2, score, topk
            )
            result = torch.zeros_like(tmp_2)
            executable_kernel(tmp_2, w2, topk_weight, topk_ids, result)
