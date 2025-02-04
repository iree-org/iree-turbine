# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn import functional as F
from torch.testing import assert_close
from typing import Any, Callable

from iree.turbine.kernel.gen import TestLaunchContext
from iree.turbine.kernel.lang.global_symbols import GLOBAL_ADDRESS_SPACE
from iree.turbine.kernel.wave.utils import (
    device_zeros,
    get_default_run_config,
    to_default_device,
)
from causal_attention_template import (get_causal_attention_kernel as
                                       get_tkw_causal_attention_kernel)

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw

torch.manual_seed(0)
torch.set_printoptions(
    linewidth=1000000,
    threshold=1000000,
    precision=3,
)

vM, vN = 10, 10
torch_o = to_default_device(torch.ones(vM, vN))
temp_mask = to_default_device(
    torch.ones(vM, vN, dtype=torch.bool).tril(diagonal=0))
torch_o.masked_fill_(temp_mask.logical_not(), float("-inf"))

M = tkl.sym.M
N = tkl.sym.N
ONE = tkl.sym.ONE
# Expose user-constraints
constraints: list[tkw.Constraint] = []
constraints += [
    tkw.HardwareConstraint(
        threads_per_wave=64,
        waves_per_block=(1, 1, 1),
        vector_shapes={
            M: 1,
            N: 1,
        },
    )
]

constraints += [tkw.WorkgroupConstraint(M, 1, 0)]
constraints += [tkw.WorkgroupConstraint(N, 1, 1)]

# WARNING: these constraints generate wrong code
# constraints += [tkw.WorkgroupConstraint(M, 2, 0)]
# constraints += [tkw.WorkgroupConstraint(N, 2, 1)]
# constraints += [tkw.WaveConstraint(M, 1)]
# constraints += [tkw.WaveConstraint(N, 1)]


### TKW Harness
def run(fun: Callable, hparams, *args) -> Any:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():  # Disable gradient calculations
            with TestLaunchContext(
                    hparams,
                    canonicalize=True,
                    # compile_config={"print_ir_after": "all"},
                    run=True,
                    run_config=get_default_run_config(),
                    run_bench=False,
                    schedule=False,
                    use_scheduling_barriers=False,
            ):
                mb = fun(*args)
                print(mb.module_op)
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=10))


@tkw.wave(constraints)
def test(o: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32]):
    i = tkw.self_index(M, tkl.i64, elements_per_thread=1)
    i = tkw.broadcast(i, target_shape=[N])
    j = tkw.self_index(N, tkl.i64, elements_per_thread=1)
    ZERO = tkl.Register[N, tkl.i64](0)
    ONE = tkl.Register[N, tkl.i64](1)
    ZEROF = tkl.Register[N, tkl.f32](0.0)
    MIN_INF = tkl.Register[N, tkl.f32](float('-inf'))
    idx = j - i - ONE
    res = tkw.select(tkw.slt(idx, ZERO), ZEROF, MIN_INF)
    val = tkw.read(o, elements_per_thread=1)
    res += val
    tkw.write(res, o, elements_per_thread=1)


o = to_default_device(torch.ones(vM, vN))
run(test, {M: vM, N: vN, ONE: 1}, o)

# print(o)
assert_close(torch_o.to(dtype=o.dtype), o, atol=2e-3, rtol=2e-3)
