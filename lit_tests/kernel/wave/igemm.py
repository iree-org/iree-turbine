# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import (
    run_test,
    get_default_compile_config,
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K = tkl.sym.BLOCK_K
BLOCK_B = tkl.sym.BLOCK_B
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0


def codegen_test_context(canonicalize: bool = False, dynamic_symbols=[]):
    bindings = {
        M: 16,
        N: 16,
        K: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        BLOCK_K: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }

    # Remove dynamic symbols from the bindings.
    for sym in dynamic_symbols:
        if sym in bindings:
            del bindings[sym]

    return tk.gen.TestLaunchContext(
        bindings, canonicalize=canonicalize, dynamic_symbols=dynamic_symbols
    )


@run_test
def test_igemm():
    n, c, h, w = 2, 640, 64, 64
    cf, hf, wf, nf = c, 3, 3, 640
    stride = 1
    padding = 0

    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (n, nf, h_out, w_out)
    out = torch.zeros(res_shape, dtype=torch.float32)

    sym = tkl.sym
    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    K = HF * WF * C
    M = SZ_OUT * N

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

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

    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # layout == "nhwc_hwcf"
    x_type = tkl.Memory[N, H, W, C, ADDRESS_SPACE, tkl.f16]
    we_type = tkl.Memory[HF, WF, C, NF, ADDRESS_SPACE, tkl.f16]
    out_type = tkl.Memory[N, H_OUT, W_OUT, NF, GLOBAL_ADDRESS_SPACE, tkl.f32]
    x = torch.permute(x, (0, 2, 3, 1)).contiguous()
    we = torch.permute(we, (2, 3, 1, 0)).contiguous()
    out = torch.permute(out, (0, 2, 3, 1)).contiguous()

    ratio_m = 2
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
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.reduction(K, init_args=[c_reg])
        def repeat(acc: tkl.Register[M, NF, tkl.f32]) -> tkl.Register[M, NF, tkl.f32]:
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

    with tk.gen.TestLaunchContext(
        {
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 64,
            BLOCK_N: 128,
            BLOCK_K: 32,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        },
        canonicalize=True,
    ):
        print(conv(x, we, out).module_op)
        # CHECK-LABEL: func @conv
        #  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index

        # Input load must be contiguous.
        #      CHECK: %{{.*}} = vector.maskedload %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], %{{.*}}, %{{.*}} : memref<2x64x64x640xf16

        # Weights are done via gather, check we are setting gather start indices to 0.
        #      CHECK: %{{.*}} = vector.gather %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : memref<3x3x640x640xf16
        #      CHECK: %{{.*}} = vector.gather %{{.*}}[%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : memref<3x3x640x640xf16
