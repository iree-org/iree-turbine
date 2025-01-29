# RUN: python %s | FileCheck %s

import pytest
from typing import Callable
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils import run_test
import torch

M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
B = tkl.sym.B
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
def test_cast():
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        a_reg = tkw.cast(a_reg, tkl.f32)
        a_reg = tkw.cast(a_reg, tkl.i8)
        a_reg = tkw.cast(a_reg, tkl.f16)
        a_reg = tkw.cast(a_reg, tkl.i16)
        a_reg = tkw.cast(a_reg, tkl.i32)
        a_reg = tkw.cast(a_reg, tkl.f32)
        a_reg = tkw.cast(a_reg, tkl.f16)
        tkw.write(a_reg, b, elements_per_thread=16)

    with codegen_test_context(canonicalize=True):
        a = torch.randn(16, 16, dtype=torch.float16)
        b = torch.zeros(16, 16, dtype=torch.float16)
        print(test(a, b).module_op)

        # CHECK:  %[[D0:.*]] = arith.extf {{.*}} : vector<16xf16> to vector<16xf32>
        # CHECK:  %[[D1:.*]] = arith.fptosi %[[D0]] : vector<16xf32> to vector<16xi8>
        # CHECK:  %[[D2:.*]] = arith.sitofp %[[D1]] : vector<16xi8> to vector<16xf16>
        # CHECK:  %[[D3:.*]] = arith.fptosi %[[D2]] : vector<16xf16> to vector<16xi16>
        # CHECK:  %[[D4:.*]] = arith.extsi %[[D3]] : vector<16xi16> to vector<16xi32>
        # CHECK:  %[[D5:.*]] = arith.sitofp %[[D4]] : vector<16xi32> to vector<16xf32>
        # CHECK:  %[[D6:.*]] = arith.truncf %[[D5]] : vector<16xf32> to vector<16xf16>
