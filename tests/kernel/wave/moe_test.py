# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
from iree.turbine.kernel.lang.global_symbols import *
from .common.utils import (
    require_cdna3,
    require_e2e,
    enable_scheduling_barriers,
    dump_generated_mlir,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.templates.moe import (
    get_gemm_kernel,
    get_silu_and_mul_kernel,
)
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.lang import DataType
import torch.nn.functional as F

torch.manual_seed(0)


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def get_wave_silu_and_mul_kernel(
    m: int,
    n: int,
    datatype: DataType,
):
    assert datatype in [torch.float16, torch.bfloat16], f"Unsupported datatype: {datatype}"
    silu_and_mul_kernel, symbols, _, _ = get_silu_and_mul_kernel(
        m,
        n,
        tkl.f16 if datatype == torch.float16 else tkl.bf16
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    options = set_default_run_config(options)
    silu_and_mul = wave_compile(options, silu_and_mul_kernel)
    return silu_and_mul


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 2, f"Expect 2D tensor input to silu, got {len(x.shape)}"
    wave_kernel = get_wave_silu_and_mul_kernel(x.shape[0], x.shape[1], x.dtype)

    d = x.shape[-1] // 2
    out = torch.zeros(x.shape[0], d, dtype=x.dtype, device=x.device)
    wave_kernel(x[..., :d], x[..., d:], out)
    return out


def get_wave_gemm_kernel(
    m: int,
    k: int,
    n: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    gemm, symbols, _, _ = get_gemm_kernel(
        m,
        k,
        n,
        mfma_variant,
        datatype,
    )
    symbols.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=symbols,
        canonicalize=True,
        run_bench=False,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
        schedule=SchedulingType.NONE,
        wave_runtime=True,
        use_scheduling_barriers=enable_scheduling_barriers,
    )
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm)
    return gemm


def torch_ref_moe(a, w1, w2, score, topk):
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
            out[mask] = silu_and_mul_ref(a[mask] @ w1[i].transpose(0, 1)) @ w2[
                i
            ].transpose(0, 1)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def softmax(x: torch.Tensor, dim: int = -1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Convert input to float32 for stable calculations
    x_float = x.to(dtype)

    # Subtract max for numerical stability
    max_vals = torch.max(x_float, dim=dim, keepdim=True).values
    x_exp = torch.exp(x_float - max_vals)

    # Compute softmax
    sum_exp = torch.sum(x_exp, dim=dim, keepdim=True)
    result = x_exp / sum_exp

    return result


def top_k(x: torch.Tensor, k: int, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    # Sort the tensor along the specified dimension
    sorted_values, sorted_indices = torch.sort(x, dim=dim, descending=True)
    # Slice the top-k elements
    top_values = sorted_values.narrow(dim, 0, k)
    top_indices = sorted_indices.narrow(dim, 0, k)
    return top_values, top_indices


def torch_tkw_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    dtype = tkl.f16 if a.dtype == torch.float16 else tkl.bf16
    rtol, atol = 1e-1, 1e-2
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            m = int(mask.sum())
            partial_out = torch.zeros(m, w2.shape[1], dtype=torch.float32, device=a.device)
            gemm_kernel_out = get_wave_gemm_kernel(
                m, # M
                w2.shape[-1], # K
                w2.shape[1], # N
                MMAType.F32_16x16x16_F16,
                dtype,
            )
            tmp = torch.zeros(m, w1.shape[1], dtype=torch.float32, device=a.device)
            gemm_kernel_tmp = get_wave_gemm_kernel(
                m, # M
                w1.shape[-1], # K
                w1.shape[1], # N
                MMAType.F32_16x16x16_F16,
                dtype,
            )
           #print(a[mask].shape)
           #print(w1[i].shape)
           #print(m, w1.shape[-1], w1.shape[1])
            gemm_kernel_tmp(a[mask], w1[i], tmp)
            tmp = tmp.to(dtype=a.dtype)
           #torch.testing.assert_close(
           #    tmp, a[mask] @ w1[i].transpose(0, 1), rtol=rtol, atol=atol, check_device=False
           #)
            lhs = silu_and_mul(tmp)
            rhs = w2[i]
            gemm_kernel_out(lhs, rhs, partial_out)
            out[mask] = partial_out.to(dtype=a.dtype)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


num_experts = [8, 64]
top_ks = [2, 6]
# Leave 1024 * 128 out for faster testing
#m_values = [1, 33, 64, 222, 1024 * 128]
m_values = [1, 33, 64, 222]
n_values = [128, 1024, 2048]
k_values = [128, 511, 1024]
dtypes = [torch.float16, torch.bfloat16]
dtypes = [torch.float16]
dtypes = [torch.bfloat16]


@require_e2e
@pytest.mark.parametrize("m", m_values)
@pytest.mark.parametrize("n", n_values)
@pytest.mark.parametrize("k", k_values)
@pytest.mark.parametrize("e", num_experts)
@pytest.mark.parametrize("topk", top_ks)
@pytest.mark.parametrize("dtype", dtypes)
def testReferenceMoe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: DataType,
):
    rtol, atol = 1e-1, 1e-2
    device = "cuda"

    if dtype == torch.float16 and k == 1024:
        pytest.skip("This combination generates NaNs")

    a = torch.rand((m, k), dtype=dtype, device=device)
    w1 = torch.rand((e, 2 * n, k), dtype=dtype, device=device)
    w2 = torch.rand((e, k, n), dtype=dtype, device=device)
    score = torch.rand((m, e), dtype=dtype, device=device)

    ref_output = torch_ref_moe(a, w1, w2, score, topk)
    tkw_output = torch_tkw_moe(a, w1, w2, score, topk)
    torch.testing.assert_close(
        tkw_output, ref_output, rtol=rtol, atol=atol
    )
