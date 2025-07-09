# RUN: python %s | FileCheck %s

import pytest
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.utils.general_utils import (
    run_test,
)
from iree.turbine.kernel._support.location_config import (
    LocationCaptureConfig,
    LocationCaptureLevel,
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
    *, loc_level: LocationCaptureLevel, use_local_scope: bool = False
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
        location_capture_config=LocationCaptureConfig(loc_level),
        use_local_scope=use_local_scope,
        canonicalize=False,
    )
    return options


@run_test
def test_location_local_scope():
    constraints = _make_constraints()
    options = _make_options(
        loc_level=LocationCaptureLevel.FILE_LINE_COL, use_local_scope=True
    )

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
    options = _make_options(
        loc_level=LocationCaptureLevel.FILE_LINE_COL, use_local_scope=False
    )

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
    options = _make_options(loc_level=LocationCaptureLevel.NONE)

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


@run_test
def test_stack_trace():
    constraints = _make_constraints()
    options = _make_options(
        loc_level=LocationCaptureLevel.STACK_TRACE, use_local_scope=True
    )

    @tkw.wave(constraints)
    def add_stack_trace(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    add_stack_trace = wave_compile(options, add_stack_trace)
    print(add_stack_trace.asm)

    # A relatively high-level check for stack trace. Verify that it is represented
    # as callsite locations and that we see not only this file, but also torch fx.
    # But not the "system frames" from iree/turbine/kernel.
    #
    # CHECK-LABEL: @add_stack_trace
    # CHECK:       arith.addf
    # CHECK-SAME:  loc(
    # CHECK-SAME:    callsite(
    # CHECK-SAME:      location.py
    # CHECK-SAME:      at callsite(
    # CHECK-SAME:        torch/fx
    # CHECK-NOT:     iree/turbine/kernel
    # CHECK:       return


@run_test
def test_stack_trace_with_system():
    constraints = _make_constraints()
    options = _make_options(
        loc_level=LocationCaptureLevel.STACK_TRACE_WITH_SYSTEM, use_local_scope=True
    )

    @tkw.wave(constraints)
    def add_stack_trace_with_system(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    add_stack_trace_with_system = wave_compile(options, add_stack_trace_with_system)
    print(add_stack_trace_with_system.asm)

    # A relatively high-level check for stack trace. Verify that it is represented
    # as callsite locations and that we see not only this file, but also torch fx
    # and "system frames" from iree/turbine/kernel.
    #
    # CHECK-LABEL: @add_stack_trace_with_system
    # CHECK:       arith.addf
    # CHECK-SAME:  loc(
    # CHECK-SAME:    callsite(
    # CHECK-SAME:      location.py
    # CHECK-SAME:      at callsite(
    # CHECK-SAME:        torch/fx
    # CHECK-SAME:     iree/turbine/kernel
    # CHECK:       return


@run_test
def test_stack_trace_dedup():
    constraints = _make_constraints()
    options = _make_options(
        loc_level=LocationCaptureLevel.STACK_TRACE, use_local_scope=False
    )

    @tkw.wave(constraints)
    def test_stack_trace_dedup(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        a_reg = tkw.read(a, elements_per_thread=16)
        res = a_reg + a_reg

    test_stack_trace_dedup = wave_compile(options, test_stack_trace_dedup)
    print(test_stack_trace_dedup.asm)

    # Ensure that stack trace location is deduplicated to avoid IR size growth.
    # In particular, check that both the read and the addition share a common
    # parent location, that of the tracer.
    #
    # CHECK-LABEL: @test_stack_trace_dedup
    # CHECK:       vector.load
    # CHECK-SAME:  loc(#[[loc_load:.+]])
    # CHECK:       arith.addf
    # CHECK-SAME:  loc(#[[loc_addf:.+]])
    # CHECK:       #[[loc_load]] = loc(callsite(#{{.*}} at #[[loc_parent:.+]])
    # CHECK:       #[[loc_addf]] = loc(callsite(#{{.*}} at #[[loc_parent]])
