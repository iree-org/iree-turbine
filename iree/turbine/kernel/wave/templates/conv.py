# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel._support.dtype import DataType
from typing import Any, Optional
from iree.turbine.kernel.lang.global_symbols import *


def get_igemm_conv2d(
    layout: str,
    n: int,
    h: int,
    w: int,
    c: int,
    hf: int,
    wf: int,
    nf: int,
    stride: int,
    input_dtype: DataType,
    output_dtype: DataType,
    mem_space: tkl.IndexSymbol = SHARED_ADDRESS_SPACE,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
    ratio_m: Optional[int] = None,
    ratio_n: Optional[int] = None,
) -> tuple["LaunchableWave", dict[tkl.IndexSymbol, Any]]:
    assert input_dtype == tkl.f16, f"Unsupported input dtype: {input_dtype}"
    assert output_dtype == tkl.f32, f"Unsupported input dtype: {output_dtype}"
    cf = c
    padding = 0  # TODO: only pad=0 is supported for now

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    # Align C dim reading pattern to be contiguous for nhwc_hwcf pattern.
    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j % C,
            H: (i % SZ_OUT) % W_OUT * stride + (j // C) % WF,
            W: (i % SZ_OUT) // W_OUT * stride + (j // C) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j % C, HF: (j // C) % WF, WF: (j // C) // WF},
        outputs={NF: i, K: j},
    )
    out_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, NF: j},
        outputs={
            N: i // SZ_OUT,
            NF: j,
            H_OUT: (i % SZ_OUT) % W_OUT,
            W_OUT: (i % SZ_OUT) // W_OUT,
        },
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    if layout == "nchw_fchw":
        x_type = tkl.Memory[N, C, H, W, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, NF, H_OUT, W_OUT, GLOBAL_ADDRESS_SPACE, output_dtype]
    elif layout == "nhwc_hwcf":
        x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, input_dtype]
        we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, input_dtype]
        out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, output_dtype]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    if block_m is None:
        block_m = 64

    if block_n is None:
        block_n = 128

    if block_k is None:
        block_k = 32

    if ratio_m is None:
        ratio_m = 2

    if ratio_n is None:
        ratio_n = 2

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / ratio_m)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N / ratio_n)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(ratio_n, ratio_m, 1),
        )
    ]

    @tkw.wave(constraints)
    def conv(
        x: x_type,
        we: we_type,
        out: out_type,
    ):
        c_reg = tkl.Register[M, NF, output_dtype](0.0)

        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[M, NF, output_dtype],
        ) -> tkl.Register[M, NF, output_dtype]:
            a_reg = tkw.read(
                x,
                mapping=x_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            b_reg = tkw.read(
                we,
                mapping=w_mapping,
                elements_per_thread=ELEMS_PER_THREAD,
            )
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            repeat, out, mapping=out_mapping, elements_per_thread=ELEMS_PER_THREAD
        )

    symbols = {
        N: n,
        C: c,
        W: w,
        H: h,
        NF: nf,
        WF: wf,
        HF: hf,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        ELEMS_PER_THREAD: 4,
        ADDRESS_SPACE: mem_space,
    }

    return conv, symbols
