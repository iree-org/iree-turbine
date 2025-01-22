# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.wave_sim import wave_sim
from iree.turbine.kernel.wave.templates.conv import get_igemm_conv2d
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils import (
    ceildiv,
    get_default_arch,
    get_default_run_config,
    get_default_scheduling_params,
    to_default_device,
    device_randn,
    device_randint,
    device_randperm,
    device_zeros,
)
import torch
from torch.testing import assert_close
import pytest
import sympy
import os
import torch
import json

require_e2e = pytest.mark.require_e2e
require_cdna3 = pytest.mark.skipif(
    "gfx94" not in get_default_arch(), reason="Default device is not CDNA3"
)
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


def xfail_unaligned(func):
    def wrapper(shape):
        if shape[-1] % 2 != 0:
            pytest.xfail("Unaligned shape is not expected to work on this test yet.")
        func(shape)

    return wrapper


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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    vmfb_file = tmp_path / "test.vmfb"
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        create_vmfb_file=vmfb_file,
        run_config=config,
    ):
        assert not os.path.exists(vmfb_file)
        test()
        assert os.path.exists(vmfb_file)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_copy(shape, request):
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b)
        assert_close(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_dynamic_copy(shape, request):
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        dynamic_symbols=(M, N),
        dynamic_symbols_map={M: shape[0], N: shape[1]},
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b)
        assert_close(a, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_read"))
def test_transpose_read(shape, request):
    run_bench = request.config.getoption("--runperf")
    shape = shape[::-1]
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_N = 1
    BLOCK_M = sympy.Max(sympy.Min(M, 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_M / wave_size

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
        res = tkw.read(a, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b)
        assert_close(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_transpose_write"))
def test_transpose_write(shape, request):
    run_bench = request.config.getoption("--runperf")
    M = tkl.sym.M
    N = tkl.sym.N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = sympy.Max(sympy.Min(N, 256), wave_size)
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(res, b, mapping=mapping, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    b = device_zeros(shape[::-1], dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b)
        assert_close(a.T, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_offset_read(shape, request):
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        offset = tkw.read(off, elements_per_thread=ELEMS_PER_THREAD)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=ELEMS_PER_THREAD,
        )
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, off, out)
        out_ref = torch.take_along_dim(a, off.to(torch.long), dim=0)
        assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_offset_read_one(shape, request):
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        offset = tkw.read(off, elements_per_thread=1)
        res = tkw.read(
            a,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=ELEMS_PER_THREAD,
        )
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = device_randint(shape[0], (shape[0], n1), dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, off, out)
        off_expanded = off.repeat_interleave(count, dim=1)[:, : shape[1]].to(torch.long)
        out_ref = torch.take_along_dim(a, off_expanded, dim=0)
        assert_close(out, out_ref)


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
    ELEMS_PER_THREAD = BLOCK_N // wave_size

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
    dynamic_symbols_map = {}

    dynamic_symbols.append(S)
    dynamic_symbols_map[S] = 0

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off, elements_per_thread=ELEMS_PER_THREAD)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
            elements_per_thread=ELEMS_PER_THREAD,
        )
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
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
    ELEMS_PER_THREAD = BLOCK_N // wave_size

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
    dynamic_symbols_map = {}

    dynamic_symbols.append(S)
    dynamic_symbols_map[S] = 0

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[S, N, ADDRESS_SPACE, tkl.f16],
        off: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        offset = tkw.read(off, elements_per_thread=ELEMS_PER_THREAD)
        offset = tkw.apply_expr(offset, lambda a: M - a - 1)
        tkw.set_symbol(S, offset)
        res = tkw.read(
            a,
            mapping=mapping,
            elements_per_thread=ELEMS_PER_THREAD,
        )
        tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    off = device_randint(shape[0], shape, dtype=torch.int32)
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map=dynamic_symbols_map,
    ):
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
        mask: tkl.Memory[M, N, ADDRESS_SPACE, tkl.i32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        cond = tkw.read(mask, elements_per_thread=ELEMS_PER_THREAD)

        @tkw.conditional(cond)
        def then():
            tkw.write(res, b, elements_per_thread=ELEMS_PER_THREAD)

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    mask = device_randint(2, shape, dtype=torch.int32)
    b = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, mask, b)
        assert_close(a * mask, b)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_copy"))
def test_offset_write(shape, request):
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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
        offset = tkw.read(off, elements_per_thread=ELEMS_PER_THREAD)
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=ELEMS_PER_THREAD,
        )

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    off = (
        device_randperm(shape[1], dtype=torch.int32)
        .reshape((1, shape[1]))
        .repeat(shape[0], 1)
    )
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, off, out)
        out_ref = torch.zeros_like(out)
        out_ref = out_ref.scatter(1, off.to(torch.long), a)
        assert_close(out, out_ref)


@require_e2e
@pytest.mark.parametrize(
    "shape", mark_shapes_xfail(get_test_shapes("test_copy"), [(111, 813)])
)
def test_offset_write_one(shape, request):
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
        offset = tkw.read(off, elements_per_thread=1)
        res = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
        tkw.write(
            res,
            b,
            mapping=mapping,
            mapping_dynamic_vals=(offset,),
            elements_per_thread=ELEMS_PER_THREAD,
        )

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float16)
    count = int(ELEMS_PER_THREAD)
    n1 = ceildiv(shape[1], count)
    off = (
        device_randperm(n1, dtype=torch.int32).reshape((1, n1)).repeat(shape[0], 1)
        * count
    )
    out = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            N1: n1,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
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

    config = get_default_run_config()

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float16)
    b = device_randn(shape, dtype=torch.float16)
    c = device_zeros((shape[0],), dtype=torch.float16)
    ref = torch.sum((a * b), dim=-1)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b, c)
        assert_close(ref, c, atol=0.1, rtol=1e-05)


@require_e2e
@pytest.mark.parametrize("shape", get_test_shapes("test_tiled_reduce_max"))
@xfail_unaligned
def test_toy_online_softmax(shape):
    M = tkl.sym.M
    N = tkl.sym.N
    wave_size = 64
    BLOCK_M = 1
    BLOCK_N = tkl.sym.BLOCK_N
    ELEMS_PER_THREAD = tkl.sym.ELEMS_PER_THREAD
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            vector_shapes={M: 1, N: BLOCK_N},
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, N, 0)]
    constraints += [tkw.TilingConstraint(N, BLOCK_N)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def test(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        b: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
        c: tkl.Memory[M, ADDRESS_SPACE, tkl.f32],
    ):
        init_max = tkl.Register[M, tkl.f32](-1e6)
        init_sum = tkl.Register[M, tkl.f32](0)

        @tkw.reduction(N, init_args=[init_max, init_sum])
        def repeat(
            partial_max: tkl.Register[M, tkl.f32],
            partial_sum: tkl.Register[M, tkl.f32],
        ) -> tkl.Register[M, tkl.f32]:
            lhs = tkw.read(a, elements_per_thread=ELEMS_PER_THREAD)
            rhs = tkw.read(b, elements_per_thread=ELEMS_PER_THREAD)
            res = lhs * rhs
            partial_max = tkw.max(res, partial_max, dim=N)
            partial_sum = tkw.sum(res, partial_sum, dim=N)
            return partial_max, partial_sum

        res_max, res_sum = repeat
        result = res_max / res_sum
        tkw.write(result, c, elements_per_thread=1)

    config = get_default_run_config()

    torch.manual_seed(1)
    a = device_randn(shape, dtype=torch.float32)
    b = device_randn(shape, dtype=torch.float32)
    c = device_zeros((shape[0],), dtype=torch.float32)
    ref_max = torch.max((a * b), dim=-1).values
    ref_sum = torch.sum((a * b), dim=-1)
    ref = ref_max / ref_sum
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            BLOCK_N: min(128, shape[1]),
            ELEMS_PER_THREAD: min(128, shape[1]) // wave_size,
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_config=config,
    ):
        test(a, b, c)
        # Assert equal does cast to boolean on torch.Tensor
        # which causes issues, hence we cast to numpy before
        # checking.
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
    # TODO: TilingConstraint doesn't work without actual reduction loop, instead
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

    config = get_default_run_config()

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, hf * wf * c)
    a = device_randn((n, c, h, w), dtype=torch.float16)
    b = device_zeros(res_shape, dtype=torch.float16)

    im2col = torch.nn.Unfold(kernel_size=(hf, wf), padding=padding, stride=stride)
    expected = im2col(a)[0, :, :].T

    with tk.gen.TestLaunchContext(
        {
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
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
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

        tkw.write(repeat, out, elements_per_thread=ELEMS_PER_THREAD)

    sim_func = wave_sim(constraints)(func)
    gpu_func = tkw.wave(constraints)(func)

    h_out = (h + 2 * padding - hf) // stride + 1
    w_out = (w + 2 * padding - wf) // stride + 1
    res_shape = (h_out * w_out * n, nf)
    out_ref = torch.zeros(res_shape, dtype=torch.float32)
    sim_func(x, we, out_ref)

    out = torch.zeros_like(out_ref)

    config = get_default_run_config()

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
            BLOCK_N: 64,
            ELEMS_PER_THREAD: 4,
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
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

perf_test = lambda *a: pytest.param(*a, marks=pytest.mark.perf_only)
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
def test_igemm_conv(n, h, w, c, hf, wf, nf, stride, mem_space, layout, request):
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

    config = get_default_run_config()

    run_bench = request.config.getoption("--runperf")
    dump_perf = request.config.getoption("--dump-perf-files-path")
    if run_bench:
        config["benchmark_batch_size"] = 10
        config["benchmark_repetitions"] = 3
        config["dump_intermediates"] = "./inter"

    if dump_perf is not None:
        perf_filename = request.node.name + ".json"
        config["benchmark_results_file"] = os.path.join(
            dump_perf, "tk_" + perf_filename
        )

    with tk.gen.TestLaunchContext(
        hyperparams,
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
        schedule=False,
    ):
        out = torch.zeros_like(out_ref)
        conv(x, we, out)
        assert_close(out, out_ref, rtol=1e-03, atol=1e-03)

        if run_bench:
            if dump_perf is not None:
                config["benchmark_results_file"] = os.path.join(
                    dump_perf, "iree_" + perf_filename
                )

            config[
                "iree_preprocessing_pass_pipeline"
            ] = "builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)"
            iree_ref = torch.zeros_like(out_ref)
            generate_iree_ref(
                "conv_2d_" + layout,
                [x, we],
                [iree_ref],
                config,
                stride=stride,
                run_bench=True,
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
    ELEMS_PER_THREAD = BLOCK_N / wave_size

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

    config = get_default_run_config()

    a = device_randn(shape, dtype=torch.float32)
    b = device_zeros(shape, dtype=torch.float16)
    with tk.gen.TestLaunchContext(
        {
            M: shape[0],
            N: shape[1],
            ADDRESS_SPACE: tkl.AddressSpace.GLOBAL_MEMORY.value,
        },
        canonicalize=True,
        run=True,
        run_bench=run_bench,
        run_config=config,
    ):
        test(a, b)
        assert_close(a.to(dtype=torch.float16), b)
