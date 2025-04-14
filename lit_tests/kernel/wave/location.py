# RUN: python %s | FileCheck %s

import pytest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)

M = tkl.sym.M
N = tkl.sym.N
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def _make_constraints() -> list[tkw.Constraint]:
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64, waves_per_block=(1, 1, 1), vector_shapes={M: 16, N: 16}
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    return constraints


def _make_options(
    *, debug_info: bool, use_local_scope: bool = False
) -> WaveCompileOptions:
    subs = {
        M: 16,
        N: 16,
        BLOCK_M: 16,
        BLOCK_N: 16,
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
    }
    options = WaveCompileOptions(
        subs=subs,
        compile_to_mlir=True,
        debug_info=debug_info,
        use_local_scope=use_local_scope,
    )
    return options


@run_test
def test_location_local_scope():
    constraints = _make_constraints()
    options = _make_options(debug_info=True, use_local_scope=True)

    @tkw.wave(constraints)
    def add_loc_local_scope(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    add_loc_local_scope = wave_compile(options, add_loc_local_scope)
    print(add_loc_local_scope.asm)

    # CHECK-LABEL: @add_loc_local_scope
    # CHECK: vector.load {{.*}} loc("{{.*}}location.py":{{[0-9]+}}
    # CHECK: arith.addf {{.*}} loc("{{.*}}location.py":{{[0-9]+}}
    #
    # CHECK: @isolated_benchmark(%{{.*}} loc("a"("{{.*}}location.py":{{[0-9]+}}{{.*}} loc("b"("{{.*}}location.py":{{[0-9]+}}


@run_test
def test_location_global_scope():
    constraints = _make_constraints()
    options = _make_options(debug_info=True, use_local_scope=False)

    @tkw.wave(constraints)
    def add_loc_global_scope(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    add_loc_global_scope = wave_compile(options, add_loc_global_scope)
    print(add_loc_global_scope.asm)

    # CHECK-DAG: #[[loc_arg:.+]] = loc("{{.*}}location.py":{{[0-9]+}}
    # CHECK-DAG: #[[loc_a:.+]] = loc("a"(#[[loc_arg]])
    # CHECK-DAG: #[[loc_b:.+]] = loc("b"(#[[loc_arg]])
    # CHECK-LABEL: @add_loc_global_scope
    # CHECK: vector.load {{.*}} loc(#[[loc_load:.+]])
    # CHECK: arith.addf {{.*}} loc(#[[loc_addf:.+]])
    # CHECK: @isolated_benchmark(%{{.*}} loc("a"(#[[loc_arg]])), %{{.*}} loc("b"(#[[loc_arg]])))
    # CHECK-DAG: #[[loc_load]] = loc("{{.*}}location.py":{{[0-9]+}}
    # CHECK-DAG: #[[loc_addf]] = loc("{{.*}}location.py":{{[0-9]+}}


@run_test
def test_no_location():
    constraints = _make_constraints()
    options = _make_options(debug_info=False)

    @tkw.wave(constraints)
    def add_no_loc_info(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    add_no_loc_info = wave_compile(options, add_no_loc_info)
    print(add_no_loc_info.asm)

    # CHECK-LABEL: @add_no_loc
    # CHECK-NOT: loc(
