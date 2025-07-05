# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os

import pytest
import sympy
import torch
from torch.testing import assert_close

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.templates.conv import get_igemm_conv2d
from iree.turbine.kernel.wave.utils.general_utils import (
    ceildiv,
    check_leaks,
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_arange,
    device_full,
    device_ones,
    device_randint,
    device_randn,
    device_randperm,
    device_zeros,
    to_default_device,
)
from iree.turbine.kernel.wave.wave_sim import wave_sim

from .common.utils import (
    param_bool,
    perf_test,
    require_cdna3,
    require_e2e,
)
from .common.shapes import get_test_shapes as get_common_test_shape


default_test_shapes = [
    (1, 27),
    (111, 813),
    (1, 128),
    (256, 64),
    (256, 128),
    (256, 256),
    (256, 1024),
]

user_specified_test_shapes = ""

test_params_path = os.environ.get("TEST_PARAMS_PATH", None)


def mark_shapes_xfail(src_shapes, xfail_shapes):
    mark = lambda *a: pytest.param(*a, marks=pytest.mark.xfail)
    return [(mark(s) if s in xfail_shapes else s) for s in src_shapes]


if test_params_path:
    with open(test_params_path, "r") as file:
        user_specified_test_shapes = json.load(file)


def get_test_shapes(test_name: str) -> list[tuple[int]]:
    if test_name in user_specified_test_shapes:
        return user_specified_test_shapes[test_name]
    return default_test_shapes


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy")[:1])
def test_dump_vmfb(shape, tmp_path, request):
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    vmfb_file = tmp_path / "test.vmfb"
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        create_vmfb_file=vmfb_file,
    )
    options = set_default_run_config(options)

    assert not os.path.exists(vmfb_file)
    test = wave_compile(options, test)
    assert os.path.exists(vmfb_file)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
@check_leaks
def test_copy(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_dynamic_copy(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_read"))
@param_bool("use_buffer_ops", "buf_ops")
def test_transpose_read(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    shape = shape[::-1]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_N = 1
    BLOCK_M = sympy.Max(sympy.Min(M, 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={N: BLOCK_N, M: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={N: i, M: j}, outputs={N: i, M: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping)
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_write"))
@param_bool("use_buffer_ops", "buf_ops")
def test_transpose_write(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(N, 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: i, N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, M, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        tkw.write(res, b, mapping=mapping)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_read(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: k, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, off.to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_read_one(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    N1 = tkl.sym.N1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N, N1: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: k, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i, N1: j // ELEMS_PER_THREAD},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N1, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = device_randint(shape[0], (shape[0], n1), dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    off_expanded = off.repeat_interleave(count, dim=1)[:, : shape[1]].to(torch.long)
    out_ref = torch.take_along_dim(a, off_expanded, dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_read_write_same(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def double(a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16]):
        res = tkw.read(a)
        double = res + res
        tkw.write(double, a)

    a = device_randn(shape, dtype=torch.float16)
    ref = a + a
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    double = wave_compile(options, double)

    double(a)
    assert_close(a, ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_set_symbol(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    S = tkl.sym.S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.

    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N, S: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
    )

    dynamic_symbols = []

    dynamic_symbols.append(S)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, off.to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_apply_expr(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    S = tkl.sym.S
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.

    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N, S: BLOCK_M},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={S: S, N: j},
        outputs={S: i, N: j},
    )

    dynamic_symbols = []

    dynamic_symbols.append(S)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        offset = tkw.apply_expr(offset, lambda a: M - a - 1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
        )
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        dynamic_symbols=dynamic_symbols,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.take_along_dim(a, (shape[0] - off - 1).to(torch.long), dim=0)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_conditional(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # TODO: Only ELEMS_PER_THREAD == 1
    # BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    BLOCK_N = wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        mask: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a)
        cond = tkw.read(mask)

        cond = tkw.apply_expr(cond, lambda a: a > 0)

        @tkw.conditional(cond)
        def then():
            tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    mask = device_randint(2, shape, dtype=torch.int32)
    b = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, mask, b)
    assert_close(a * mask, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_write(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: k},
        dynamic_val_mappings={M: i, N: j},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(a)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )

    a = device_randn(shape, dtype=torch.float16)
    off = (
        device_randperm(shape[1], dtype=torch.int32)
        .reshape((1, shape[1]))
        .repeat(shape[0], 1)
    )
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.zeros_like(out)
    out_ref = out_ref.scatter(1, off.to(torch.long), a)
    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize(
    "shape", mark_shapes_xfail(get_test_shapes("test_copy"), [(111, 813)])
)
@param_bool("use_buffer_ops", "buf_ops")
def test_offset_write_one(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    N1 = tkl.sym.N1
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N, N1: 1},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.dynamic_val(0)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i, N: k + j % ELEMS_PER_THREAD},
        dynamic_val_mappings={M: i, N1: j // ELEMS_PER_THREAD},
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N1, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off)
        res = tkw.read(a)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
        )

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = (
        device_randperm(n1, dtype=torch.int32).reshape((1, n1)).repeat(shape[0], 1)
        * count
    )
    out = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, off, out)
    out_ref = torch.zeros_like(out)
    off_expanded = off.repeat_interleave(count, dim=1)
    off_expanded = off_expanded + to_default_device(
        torch.arange(count, dtype=torch.int32)
    ).reshape((1, count)).repeat(shape[0], n1)
    off_expanded = off_expanded[:, : shape[1]].to(torch.long)
    out_ref = out_ref.scatter(1, off_expanded.to(torch.long), a)

    assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_reduce_sum"))
def test_reduce_sum(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ELEMS_PER_THREAD = BLOCK_N // wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.sum(res, dim=N)
        tkw.write(res, c, elements_per_thread=1)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros((shape[0],), dtype=torch.float16)
    ref = torch.sum((a * b), dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=0.1, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_common_test_shape("test_block_reduce"))
def test_block_reduce_sum(shape, request):
    run_bench = request.config.getoption("--runperf")
    round_to_divisible = lambda src, denom: sympy.ceiling(src / denom) * denom
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    num_waves = 4
    BLOCK_M = 1

    # Distribute N dim across num_waves, and pad to disivible by wave_size.
    ELEMS_PER_WAVE = round_to_divisible(sympy.ceiling(N / num_waves), wave_size)
    # Minimum number of elems per wave should be size of wave.
    ELEMS_PER_WAVE = sympy.Max(ELEMS_PER_WAVE, wave_size)
    BLOCK_N = ELEMS_PER_WAVE * num_waves
    ELEMS_PER_THREAD = ELEMS_PER_WAVE // wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(num_waves, 1, 1),
            vector_shapes={M: 1, N: ELEMS_PER_WAVE},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, ELEMS_PER_WAVE)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs * rhs
        res = tkw.sum(res, dim=N, block=True)
        tkw.write(res, c, elements_per_thread=1)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_randn(shape, dtype=torch.float32)
    c = device_zeros((shape[0],), dtype=torch.float32)
    ref = torch.sum((a * b), dim=-1)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=2e-5, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_tiled_reduce_max"))
def test_toy_online_softmax(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_max = tkl.Register[M, tkl.f32](-1e6)
        init_sum = tkl.Register[M, tkl.f32](0)

        @tkw.iterate(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            lhs = tkw.read(a)
            rhs = tkw.read(b)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            partial_sum = tkw.sum(res, partial_sum, dim=N)
            return partial_max, partial_sum

        res_max, res_sum = repeat
        result = res_max / res_sum
        tkw.write(result, c)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_randn(shape, dtype=torch.float32)
    c = device_zeros((shape[0],), dtype=torch.float32)
    ref_max = torch.max((a * b), dim=-1).values
    ref_sum = torch.sum((a * b), dim=-1)
    ref = ref_max / ref_sum
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_N: max(min(128, shape[1]), wave_size),
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=False,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c, atol=0.1, rtol=1e-4)


@require_e2e
def test_im2col(request):
    run_bench = request.config.getoption("--runperf")
    # TODO: we don't support unaligned access at the moment so all sizes must
    # be aligned to WG/Wave sizes, c * hw * wf == 8 and number of windows == 64.
    n, c, h, w = 1, 2, 9, 9  # Image.
    cf, hf, wf = c, 2, 2  # Filters.
    padding = 0
    stride = 1

    sym = tkl.sym
    ADDRESS_SPACE = sym.ADDRESS_SPACE

    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF

    H_OUT = (H + 2 * padding - HF) // stride + 1
    W_OUT = (W + 2 * padding - WF) // stride + 1
    SZ_OUT = H_OUT * W_OUT

    # K = HF * WF * C
    # M = SZ_OUT * N
    M = sym.M
    K = sym.K

    # We unroll K dimension according to ELEMS_PER_THREAD value.
    # i.e. for K==8 we will have 2 vector.gather's.
    # Each WG will process 64 windows.
    wave_size = 64
    BLOCK_K = hf * wf * c
    BLOCK_M = 64
    ELEMS_PER_THREAD = 4

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
            W: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
        },
        outputs={M: i, K: j},
    )

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, K: ELEMS_PER_THREAD},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(K, BLOCK_K, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(K, BLOCK_K)]
    # TODO: TilingConstraint doesn't work without actual iterate loop, instead
    # we treat K as WG '1' dimension, but corresponding WG size will be always
    # equal to 1.
    # constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, hf * wf * c)
    a = device_randn((n, c, h, w), dtype=torch.float16)
    b = device_zeros(res_shape, dtype=torch.float16)

    im2col = torch.nn.Unfold(kernel_size=(hf, wf), padding=padding, stride=stride)
    expected = im2col(a)[0, :, :].T

    options = WaveCompileOptions(
        subs={
            N: n,
            C: c,
            W: w,
            H: h,
            WF: wf,
            HF: hf,
            M: res_shape[0],
            K: res_shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(b, expected)


# TODO: Fix test for CDNA2. CDNA2 seem to have worse accuracy, atol=0.0094, rtol=10.2405
@require_e2e
def test_im2col_mma(request):
    run_bench = request.config.getoption("--runperf")
    # igemm without final col2im
    n, c, h, w = 1, 4, 9, 9  # Image.
    nf, cf, hf, wf = 64, c, 2, 2  # Filters.
    padding = 0  # TODO: only pad=0 is supported for now
    stride = 1

    x = torch.randn(n, c, h, w, dtype=torch.float16)
    we = torch.randn(nf, cf, hf, wf, dtype=torch.float16)

    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach()

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

    x_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={
            N: i // SZ_OUT,
            C: j // (HF * WF),
            H: (i % SZ_OUT) % W_OUT * stride + (j % (HF * WF)) % WF,
            W: (i % SZ_OUT) // W_OUT * stride + (j % (HF * WF)) // WF,
        },
        outputs={M: i, K: j},
    )
    w_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={NF: i % NF, C: j // (HF * WF), HF: j % WF, WF: (j % (HF * WF)) // WF},
        outputs={NF: i, K: j},
    )

    # Workgroup tile sizes
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    # BLOCK_K = tkl.sym.BLOCK_K
    BLOCK_K = K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(NF, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(NF, BLOCK_N)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            # vector_shapes={NF: 1, M: BLOCK_M, K: ELEMS_PER_THREAD},
        )
    ]

    def func(
        x: tkl.Memory[N, C, H, W, ADDRESS_SPACE, tkl.f16],
        we: tkl.Memory[NF, C, HF, WF, ADDRESS_SPACE, tkl.f16],
        out: tkl.Memory[M, NF, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        c_reg = tkl.Register[M, NF, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg])
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

        tkw.write(repeat, out, elements_per_thread=ELEMS_PER_THREAD)

    sim_func = wave_sim(constraints)(func)
    gpu_func = tkw.wave(constraints)(func)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, nf)
    out_ref = torch.zeros(res_shape, dtype=torch.float32)
    sim_func(x, we, out_ref)

    out = torch.zeros_like(out_ref)

    options = WaveCompileOptions(
        subs={
            N: n,
            C: c,
            W: w,
            H: h,
            NF: nf,
            WF: wf,
            HF: hf,
            BLOCK_M: 64,
            BLOCK_N: 64,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    gpu_func = wave_compile(options, gpu_func)

    x = x.to("cuda")
    we = we.to("cuda")
    out = out.to("cuda")
    gpu_func(x, we, out)
    assert_close(out, out_ref, rtol=1e-05, atol=1e-05, check_device=False)


_igemm_cases = [
    (1, 5, 5, 10, 2, 2, 2, 2),
    (2, 5, 5, 3, 2, 2, 1, 1),
    (4, 5, 5, 10, 2, 2, 2, 1),
    (2, 5, 5, 10, 2, 2, 1, 1),
    (2, 5, 5, 10, 2, 2, 2, 1),
    (1, 5, 5, 10, 2, 2, 16, 1),
    (1, 5, 5, 10, 2, 2, 1, 2),
    (1, 5, 5, 4, 2, 2, 2, 1),
    (4, 5, 5, 10, 2, 2, 2, 3),
    (4, 5, 5, 10, 2, 2, 1, 3),
    (4, 5, 5, 10, 2, 2, 16, 2),
    (1, 5, 5, 3, 2, 2, 2, 2),
    (4, 5, 5, 10, 2, 2, 16, 1),
    (4, 5, 5, 4, 2, 2, 16, 1),
    (2, 5, 5, 4, 2, 2, 1, 3),
    (2, 5, 5, 4, 2, 2, 2, 1),
    (1, 5, 5, 10, 2, 2, 16, 3),
    (4, 5, 5, 4, 2, 2, 16, 2),
    (4, 5, 5, 10, 2, 2, 2, 1),
    (4, 5, 5, 3, 2, 2, 1, 1),
    (4, 5, 5, 4, 2, 2, 2, 1),
    (4, 5, 5, 3, 2, 2, 2, 1),
    (2, 5, 5, 1, 2, 2, 1, 3),
    (2, 5, 5, 4, 2, 2, 2, 1),
    (2, 5, 5, 10, 2, 2, 16, 1),
    (1, 5, 5, 1, 3, 3, 1, 1),
]

validation_test = lambda *a: pytest.param(*a, marks=pytest.mark.validate_only)

_igemm_cases += [
    perf_test(2, 128, 128, 16, 3, 3, 320, 1),
    perf_test(2, 128, 128, 320, 1, 1, 640, 1),
    perf_test(2, 128, 128, 320, 1, 1, 960, 1),
    perf_test(2, 128, 128, 320, 3, 3, 16, 1),
    perf_test(2, 128, 128, 320, 3, 3, 320, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 1920, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 2560, 1),
    perf_test(2, 32, 32, 1280, 1, 1, 640, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 1280, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 1920, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 2560, 1),
    perf_test(2, 32, 32, 1280, 3, 3, 640, 1),
    perf_test(2, 32, 32, 640, 3, 3, 640, 1),
    perf_test(2, 64, 64, 320, 3, 3, 320, 1),
    perf_test(2, 64, 64, 640, 1, 1, 1280, 1),
    perf_test(2, 64, 64, 640, 1, 1, 1920, 1),
    perf_test(2, 64, 64, 640, 1, 1, 320, 1),
    perf_test(2, 64, 64, 640, 1, 1, 960, 1),
    perf_test(2, 64, 64, 640, 3, 3, 320, 1),
    perf_test(2, 64, 64, 640, 3, 3, 640, 1),
]

_mem_spaces = [
    pytest.param(GLOBAL_ADDRESS_SPACE, id="global", marks=pytest.mark.validate_only),
    pytest.param(SHARED_ADDRESS_SPACE, id="shared"),
]

_layouts = [
    pytest.param("nchw_fchw", marks=pytest.mark.validate_only),
    pytest.param("nhwc_hwcf"),
]


@require_e2e
@require_cdna3
@pytest.mark.parametrize("n, h, w, c, hf, wf, nf, stride", _igemm_cases)
@pytest.mark.parametrize("mem_space", _mem_spaces)
@pytest.mark.parametrize("layout", _layouts)
@param_bool("use_buffer_ops", "buf_ops")
def test_igemm_conv(
    n, h, w, c, hf, wf, nf, stride, mem_space, layout, use_buffer_ops, request
):
    cf = c
    padding = 0  # TODO: only pad=0 is supported for now

    torch.manual_seed(1)
    x = device_randn(n, c, h, w, dtype=torch.float16)
    we = device_randn(nf, cf, hf, wf, dtype=torch.float16)

    convRef = torch.nn.Conv2d(c, nf, hf, stride=stride, padding=padding, bias=False)
    convRef.weight = torch.nn.Parameter(we)
    out_ref = convRef(x).detach().to(torch.float32)

    if layout == "nchw_fchw":
        pass  # Nothing
    elif layout == "nhwc_hwcf":
        x = torch.permute(x, (0, 2, 3, 1)).contiguous()
        we = torch.permute(we, (2, 3, 1, 0)).contiguous()
        out_ref = torch.permute(out_ref, (0, 2, 3, 1)).contiguous()
    else:
        raise ValueError(f"Invalid layout: {layout}")

    conv, hyperparams = get_igemm_conv2d(
        layout=layout,
        n=n,
        h=h,
        w=w,
        c=c,
        hf=hf,
        wf=wf,
        nf=nf,
        stride=stride,
        input_dtype=tkl.f16,
        output_dtype=tkl.f32,
        mem_space=mem_space,
    )
    hyperparams.update(get_default_scheduling_params())

    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")

    perf_filename = request.node.name + ".json"
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
        benchmark_batch_size=10,
        benchmark_repetitions=3,
        benchmark_results_file=(
            os.path.join(dump_perf, "tk_" + perf_filename) if dump_perf else None
        ),
        dump_intermediates="./inter",
    )
    options = set_default_run_config(options)
    conv = wave_compile(options, conv)

    out = torch.zeros_like(out_ref)
    conv(x, we, out)
    assert_close(out, out_ref, rtol=1e-03, atol=1e-03)

    if run_bench:
        if dump_perf is not None:
            options.benchmark_results_file = os.path.join(
                dump_perf, "iree_" + perf_filename
            )

        options.iree_preprocessing_pass_pipeline = "builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)"
        iree_ref = torch.zeros_like(out_ref)
        generate_iree_ref(
            "conv_2d_" + layout,
            [x, we],
            [iree_ref],
        )


@require_e2e
@pytest.mark.parametrize("shape", [(256, 64)])
def test_cast(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    # Each workgroup works on single row of input data, and rows are further
    # split into blocks of size up to 256. We have single wave per WG,
    # and with default wave size of 64, each thread is operating on up to 4
    # elements.
    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        res = tkw.cast(res, tkl.f16)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    a = device_randn(shape, dtype=torch.float32)
    b = device_zeros(shape, dtype=torch.float16)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    assert_close(a.to(dtype=torch.float16), b)


@require_e2e
@require_cdna3
@pytest.mark.parametrize(
    "shape", get_test_shapes("test_copy")[:2]
)  # testing on just two shapes.
@pytest.mark.parametrize(
    "tkl_dtype, torch_dtype, arg_vals",
    [  # arg_vals are c, d, e, res
        (tkl.i32, torch.int32, (1, 2, 1, 3)),
        (tkl.f32, torch.float32, (1.0, 2.0, 1.0, 3.0)),
    ],
    ids=["i32", "f32"],
)
@param_bool("use_wave_runtime", "wr", [False, True])
def test_scalar_codegen(
    shape, tkl_dtype, torch_dtype, arg_vals, request, use_wave_runtime
):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl_dtype],
        c: tkl_dtype,  # type: ignore
        d: tkl_dtype,  # type: ignore
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl_dtype],
    ):
        res = tkw.read(a)
        c = tkw.broadcast(c, target_shape=[M, N])
        d = tkw.broadcast(d, target_shape=[M, N])
        e = tkl.Register[M, N, tkl_dtype](arg_vals[2]) * c
        res = res + e + d
        tkw.write(res, b)

    a = device_zeros(shape, dtype=torch_dtype)
    b = device_zeros(shape, dtype=torch_dtype)
    scalar_c = arg_vals[0]
    scalar_d = arg_vals[1]

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        wave_runtime=use_wave_runtime,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, scalar_c, scalar_d, b)

    expected_val = torch.full_like(b, arg_vals[3])
    if tkl.f32 == tkl_dtype and not use_wave_runtime:
        # TODO: iree runtime doesn't work with f32.
        with pytest.raises(Exception):
            assert_close(b, expected_val)
    else:
        assert_close(b, expected_val)


#  This kernel copies of data from a into b if tid.x < threshold.
#  This test is important to ensure:
#  1. tkw.Scalar can handle index expressions correctly.
#  2. Scalars in Wave can be used for comparison/binaryOps
#     as well as on select ops.
@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_scalar_cond_copy(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    # Tile size cannot be dynamic, so we use a fixed value here.
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    # multiple of 4 to prevent to not require iota mask.
    # e.g if each thread has 4 values, and thresh is 10.
    # then t0 = [0, 1, 2, 3], t1 = [4, 5, 6, 7], t2 = [8, 9, 10, 11],
    # since t2_tidx_expr = 2 * 4 = 8, which is less than thresh, then
    # [10, 11] will also not be masked. To fix we'd need the iota mask
    # but not the main point of this test.
    thresh_value = 12
    tidx_expr = THREAD_0 * (BLOCK_N // wave_size) + (WORKGROUP_0 * BLOCK_N)

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        zero = tkw.scalar(0.0, tkl.f16)
        one = tkw.scalar(1.0, tkl.f16)

        tid = tkw.scalar(tidx_expr, tkl.i32)
        thresh = tkw.scalar(thresh_value, tkl.i32)

        mask = tkw.select(tid < thresh, one, zero)
        mask_broadcast = tkw.broadcast(mask, target_shape=[M, N])

        a_reg = tkw.read(a)
        res = a_reg * mask_broadcast
        tkw.write(res, b)

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, b)
    # Check for data from tid.x < threshold
    assert_close(a[:, :thresh_value], b[:, :thresh_value])

    # Check for data from tid.x >= threshold
    ref_zeros = device_zeros([shape[0], shape[1] - thresh_value])
    assert_close(ref_zeros, b[:, thresh_value:], check_dtype=False)


@require_e2e
@pytest.mark.parametrize(
    "shape",
    [
        (1, 27),
        (1, 64),
        (51, 64),
        (128, 64),
        (1, 256),
        (1, 512),
        (64, 500),
    ],
)
def test_scanop_cumsum(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    num_warps = 1
    BLOCK_M = 1
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ELEMS_PER_THREAD = (BLOCK_N // num_warps) // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N // num_warps},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        res = tkw.cumsum(lhs, dim=N)
        tkw.write(res, c)

    torch.manual_seed(1)
    input = device_randint(low=1, high=5, size=shape, dtype=torch.int32)
    output = device_zeros(shape, dtype=torch.int32)
    torch_ref = torch.cumsum((input), dim=-1, dtype=torch.int32)
    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(input, output)
    assert_close(torch_ref, output, atol=1e-03, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_vector_add")[:2])
@param_bool("use_buffer_ops", "buf_ops")
def test_vector_add(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(shape[1], 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N // wave_size

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: BLOCK_N},
        )
    ]

    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
        res = lhs + rhs
        tkw.write(res, c, elements_per_thread=ELEMS_PER_THREAD)

    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros(shape, dtype=torch.float16)
    ref = a + b

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)

    test = wave_compile(options, test)

    test(a, b, c)
    assert_close(ref, c)


@require_e2e
@pytest.mark.parametrize("shape", [(2, 128), (256, 1024)])
@param_bool("use_buffer_ops", "buf_ops")
def test_fused_softmax(shape, use_buffer_ops, request):

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, N: N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
    ):
        val = tkw.read(a)
        row_max = tkw.max(val, dim=N)
        row_max_bcast = tkw.broadcast(row_max, [M, N])
        val -= row_max_bcast
        val = tkw.exp(val)
        denominator = tkw.sum(val, dim=N)
        denom_broadcast = tkw.broadcast(denominator, [M, N])
        val = val / denom_broadcast
        tkw.write(val, b)

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_zeros(shape, dtype=torch.float32)

    ref = torch.softmax(a, dim=1)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=False,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, b)
    assert_close(ref, b)


@require_e2e
@pytest.mark.parametrize("shape", [(2, 64)])
@param_bool("use_buffer_ops", "buf_ops")
def test_atomic_min(shape, use_buffer_ops, request):
    run_bench = request.config.getoption("--runperf")

    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 2
    BLOCK_N = 64
    num_waves = 2

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, num_waves, 1),
            vector_shapes={
                M: int(BLOCK_M / num_waves),
                N: BLOCK_N,
            },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, sympy.Integer(1))]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: sympy.Integer(0), N: j},
        outputs={M: i, N: j},
    )
    read_mapping = tkw.IndexMapping(
        num_iterators=2, inputs={M: sympy.Integer(0), N: j}, outputs={M: i, N: j}
    )

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
    ):
        res = tkw.read(a)
        # We allocate a buffer of (1,BLOCK_N) shape to perform reduction across
        # waves. Inputs are distributed with (1,BLOCK_N) shape across each wave
        # and performs atomic min operation on this shared memory space. Mapping
        # attribute to atomic_min op is utilized to access the same shared memory
        # from different waves.
        shmem = tkw.allocate(
            shape=(M, N),
            distributed_shape=(1, BLOCK_N),
            dtype=tkl.i32,
        )
        inf_reg = tkl.Register[M, N, tkl.i32](1e6)
        tkw.write(inf_reg, shmem)
        res = tkw.atomic_min(res, shmem, mapping=mapping)
        res = tkw.read(shmem, mapping=read_mapping)
        tkw.write(res, c)

    a = device_randint(low=0, high=10, size=shape, dtype=torch.int32)
    b = torch.min(a, dim=0)[0].detach()
    c = device_zeros(size=shape, dtype=torch.int32)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
        use_buffer_load_ops=use_buffer_ops,
        use_buffer_store_ops=use_buffer_ops,
        minimize_shared_allocs=False,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)
    test(a, c)
    assert_close(c[0, :], b)
    assert_close(c[1, :], b)


@require_e2e
@pytest.mark.parametrize("shape", [(48, 4, 128)])
def test_self_index(shape, request):
    run_bench = request.config.getoption("--runperf")

    M = tkl.sym.M
    K = tkl.sym.K
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = shape[0]
    BLOCK_N = sympy.ceiling(N / wave_size) * wave_size

    constraints = [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=wave_size,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: BLOCK_M, K: 0, N: BLOCK_N},
        )
    ]

    # This kernel contains reduction + self_index.
    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, K, N, ADDRESS_SPACE, tkl.i32],
        result_self_index: tkl.Memory[N, GLOBAL_ADDRESS_SPACE, tkl.i32],
    ):
        input = tkw.read(a)
        # reduction will update the indices.
        input_sum = tkw.sum(input, dim=N)
        self_idx = tkw.self_index(N, dtype=tkl.i32)
        tkw.write(self_idx, result_self_index)

    torch.manual_seed(0)
    ref = device_arange(128, dtype=torch.int32)
    a = device_ones(shape, dtype=torch.int32)
    result_self_index = device_zeros(shape[2], dtype=torch.int32)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            K: shape[1],
            N: shape[2],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run_bench=run_bench,
    )
    options = set_default_run_config(options)
    test = wave_compile(options, test)

    test(a, result_self_index)
    assert_close(ref, result_self_index)
