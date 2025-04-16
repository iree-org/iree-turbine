# RUN: python %s | FileCheck %s

import copy
import logging
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.testing import assert_close

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.compiler.ir import Context, Location, Module
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.wave import LaunchableWave
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
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
        topk_weights, topk_ids = torch.topk(score, topk)
        topk_weights = topk_weights.view(-1)
        topk_ids = topk_ids.view(-1)
        for i in range(w1.shape[0]):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = silu_and_mul(
                    a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
        return (out.view(B, -1, w2.shape[1]) *
                topk_weights.view(B, -1, 1).to(out.dtype)).sum(dim=1)

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

Instead we'll use TOPK, B, D2 for the output tensor because TK does not like
1-broadcast dimensions in another form.
RESULT[B, TOPK, D2] = TMP_3[TOPK, B, D2] * topk_score[B, TOPK]
"""

# Static shapes
vB = 16
vN = 32
vD1 = 64
vD2 = 128
vTOPK = 5
vE = 256

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


def lit_harness(build_constraints_fun, get_kernel_fun, test_config):
    with tk.gen.TestLaunchContext(
        test_config["static_symbols"] if "static_symbols" in test_config else {}
    ):
        executable_kernel = get_kernel_fun(build_constraints_fun(test_config))
        trace: CapturedTrace = executable_kernel._trace()
        options = WaveCompileOptions(
            subs=test_config["static_symbols"]
            if "static_symbols" in test_config
            else {},
        )
        options.print_ir_before = ["all"]
        idxc: IndexingContext = IndexingContext.current()
        graph_passes = executable_kernel.build_full_pass_pipeline(trace, options)
        for p in graph_passes:
            try_apply_pass(p, trace, print_ir_before=["all"])

        executable_kernel.infer_grid_shape(idxc)

        options = set_default_compile_config(options)
        options.canonicalize = True
        with Context() as context:
            (
                mb,
                trace,
                exe,
                kernel_sig,
                entrypoint_name,
            ) = executable_kernel.compile_to_mlir(trace, context, options=options)
            print(mb.module_op)


def build_block_constraints(test_config) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [
        tkw.WorkgroupConstraint(B, BLOCK_B, 2),
        tkw.WaveConstraint(B, 1),
    ]
    constraints += [
        tkw.WorkgroupConstraint(TOPK, BLOCK_TOPK, 1),
        tkw.WaveConstraint(TOPK, 1),
    ]
    constraints += [
        tkw.WorkgroupConstraint(D2, BLOCK_D2, 0),
        tkw.WaveConstraint(D2, 1),
    ]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            waves_per_block=(1, 1, 1),
            threads_per_wave=64,
            # One must always specify mma_type or vector_shapes.
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes=test_config["vector_shapes"]
            if "vector_shapes" in test_config
            else {},
        )
    ]
    return constraints


def create_test_config(mma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16):
    # fmt: off
    return {
        "static_symbols": {
            ### Problem sizes.
            N: vN,
            # D1: vD1,
            # z, y, x
            B:       vB, # z
            TOPK: vTOPK, # y
            D2:     vD2, # x
            ### Block sizes.
            # tiling
            BLOCK_N: 16,
            # z, y, x
            BLOCK_B:   16, # z
            BLOCK_TOPK: 1, # y
            # Note: if BLOCK_D2 < 128, we get error : HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the largest legal address. code: 0x29
            BLOCK_D2: 128, # x
            ### L/S sizes (ideally omitted).
            LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mma_variant),
            STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mma_variant),
        },
        # Need to specify vector_shape explicitly because somehow this does
        # not get propagated.
        "vector_shapes": {
            # N: 1,
            # D1: 16,  # TODO: connected to MFMA op type
            # z, y, x
            B:    16, # z
            TOPK:  1, # y
            D2:  16, # x
        },
        "canonicalize": {True},
    }
    # fmt: on


i, j, k = [tkw.IndexMapping.iterator(i) for i in range(3)]
d0 = tkw.IndexMapping.dynamic_val(0)
offset_mapping_w2 = tkw.IndexMapping(
    num_iterators=3,
    inputs={TOPK: d0, D2: j, N: k},
    outputs={TOPK: i, D2: j, N: k},
    dynamic_val_mappings=({B: i, TOPK: j}),
)

# Note: W2 really has torch.Tensor.shape [E, D2, N] but we want to index it with
# indices [B, TOPK, D2, N].
# We don't want to introduce index E because we'd get the cartesian product
# [E, B, TOPK], which is not what we want.
# So we just use TOPK to index into W2 (alternatively we could use B).
w2_layout = tkl.MemoryLayout(
    shape=(
        vE,
        D2,
        N,
    )
)


def fused_moe_kernel(
    TMP_2: tkl.Memory[B, TOPK, N, ADDRESS_SPACE, tkl.f16],
    W2: tkl.Memory[TOPK, D2, N, ADDRESS_SPACE, tkl.f16, w2_layout],
    TOPK_IDS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.i64],
    TOPK_WEIGHTS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.f32],
    RESULT: tkl.Memory[B, TOPK, D2, ADDRESS_SPACE, tkl.f32],
):
    res_reg = tkl.Register[TOPK, B, D2, tkl.f32](0.0)

    # fmt: off
    @tkw.reduction(N, init_args=[res_reg])
    def repeat(acc: tkl.Register[TOPK, B, D2, tkl.f32]) -> tkl.Register[TOPK, B, D2, tkl.f32]:
        ###
        # TMP_3[TOPK, B, D2] = TMP_2[TOPK, B, N:]
        #   @ W2[subset(E by topk_idx[B, TOPK]), D2, N].transpose(0, 1)
        ###
        # elements_per_thread=LOAD_ELEMS_PER_THREAD, automatically derived by the
        # system from the mma op for all reads below.
        tmp_2_reg = tkw.read(TMP_2,)                            # : [TOPK, B, N]
        expert_id = tkw.read(TOPK_IDS,)                         # : [B, TOPK]
        w2_reg = tkw.read(
            W2,
            mapping=offset_mapping_w2,
            mapping_dynamic_vals=(expert_id,),
        )                                                       # : [TOPK, D2, N] but indexed as [E=(B, TOPK), D2, N] and E expert_id
        acc = tkw.mma(tmp_2_reg, w2_reg, acc)                   # : [TOPK, B, N] * [TOPK, D2, N] -> [TOPK, B, D2]
        return acc

    res = repeat

    ### res combined with read and automatic mma inference does not work.
    ### errors out with "ValueError: index 16 is out of bounds for array with size 16"
    ### Instead roundtrip through memory.
    ### elements_per_thread=STORE_ELEMS_PER_THREAD, automatically derived by the
    ### system from the mma op above.
    tkw.write(res, RESULT,)                                                 # : [B, TOPK, D2]

    ###
    # RESULT[B, TOPK, D2] = TMP_3[TOPK, B, D2] * topk_score[B, TOPK]
    ###
    ### Elementwise part, use elements_per_thread=4
    res = tkw.read(RESULT, elements_per_thread=4,)                          # : [B, TOPK, D2]
    topk_weights = tkw.read(TOPK_WEIGHTS, elements_per_thread=4,)           # : [B, TOPK]
    topk_weights = tkw.broadcast(topk_weights, target_shape=[B, TOPK, D2])  # : [B, TOPK, D2]
    res = res * tkw.cast(topk_weights, tkl.f32)                             # : [B, TOPK, D2]
    tkw.write(res, RESULT, elements_per_thread=4,)                          # : [B, TOPK, D2]
    # fmt: on


# Note: W2 really has torch.Tensor.shape [E, D2, N] but we want to index it
# with indices [B, TOPK, D2, N].
# We don't want to introduce index E because we'd get the cartesian product
# [E, B, TOPK], which is not what we want.
# So we just use TOPK to index into W2 (alternatively we could use B).
def get_fused_moe_kernel(constraints):
    @tkw.wave(constraints)
    def fused_moe_kernel_executable(
        TMP_2: tkl.Memory[B, TOPK, N, ADDRESS_SPACE, tkl.f16],
        W2: tkl.Memory[TOPK, D2, N, ADDRESS_SPACE, tkl.f16, w2_layout],
        TOPK_IDS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.i64],
        TOPK_WEIGHTS: tkl.Memory[B, TOPK, ADDRESS_SPACE, tkl.f32],
        RESULT: tkl.Memory[B, TOPK, D2, ADDRESS_SPACE, tkl.f32],
    ):
        return fused_moe_kernel(TMP_2, W2, TOPK_IDS, TOPK_WEIGHTS, RESULT)

    return fused_moe_kernel_executable


def silu_and_mul(x: torch.Tensor):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def torch_naive_moe(a, w1, w2, score, topk, result_dtype):
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
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(result_dtype)
    # .sum(dim=1)


def torch_naive_moe_step_1(a, w1, w2, score, topk):
    # where
    # a: Tensor[B, D1],
    # W1: Tensor[E, 2 * N, D1],
    # W2: Tensor[E, D2, N],
    # score: Tensor[B, E], topk: int
    # out: Tensor[B, TOPK, D2]
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out_tmp = torch.zeros(B * topk, w1.shape[1] // 2, dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk)
    topk_weights = topk_weights.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out_tmp[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1))
    return topk_weights, topk_ids, out_tmp


def torch_naive_moe_step_2(w1, w2, topk_weights, topk_ids, out_tmp):
    out = torch.zeros(
        out_tmp.shape[0], w2.shape[1], dtype=out_tmp.dtype, device=out_tmp.device
    )
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = out_tmp[mask] @ w2[i].transpose(0, 1)
    return out


def torch_naive_moe_step_3(a, w2, topk_weights, out, result_dtype):
    B, D = a.shape
    return out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(result_dtype)
    # .sum(dim=1)


if __name__ == "__main__":

    @run_test
    def static_correct_1():
        test_config = copy.deepcopy(create_test_config())
        lit_harness(build_block_constraints, get_fused_moe_kernel, test_config)

        a = torch.randn(vB, vD1, dtype=torch.float16).cuda()
        w1 = torch.randn(vE, 2 * vN, vD1, dtype=torch.float16).cuda()
        w2 = torch.randn(vE, vD2, vN, dtype=torch.float16).cuda()
        score = torch.randn(vB, vE).cuda()
        topk = vTOPK
        reference = torch_naive_moe(a, w1, w2, score, topk, result_dtype=torch.float32)

        cmp_params = dict(atol=3e-3, rtol=3e-3, check_dtype=False)
        ref_full_1 = torch_naive_moe(a, w1, w2, score, topk, result_dtype=torch.float32)
        topk_weights, topk_ids, out_tmp = torch_naive_moe_step_1(a, w1, w2, score, topk)
        out_tmp_2 = torch_naive_moe_step_2(w1, w2, topk_weights, topk_ids, out_tmp)
        ref_full_2 = torch_naive_moe_step_3(
            a, w2, topk_weights, out_tmp_2, result_dtype=torch.float32
        )
        assert_close(ref_full_1, ref_full_2, **cmp_params)

        from torch.profiler import profile, ProfilerActivity

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            test_config = create_test_config()
            executable_kernel = get_fused_moe_kernel(
                build_block_constraints(test_config)
            )

            options = WaveCompileOptions(
                subs=test_config["static_symbols"]
                if "static_symbols" in test_config
                else {},
            )
            options = set_default_compile_config(options)
            options.canonicalize = True
            options = set_default_run_config(options)
            executable_kernel = wave_compile(options, executable_kernel)

            result = torch.zeros_like(ref_full_1).cuda()
            executable_kernel(
                out_tmp.view(vB, vTOPK, vN),
                w2,
                topk_ids.view(vB, vTOPK),
                topk_weights.view(vB, vTOPK),
                result,
            )
            assert_close(result, ref_full_1, **cmp_params)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
