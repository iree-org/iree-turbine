# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pdb import run
import pytest
import torch
from typing import Callable

from iree.turbine.kernel._support.tracing import TestLaunchContext
from iree.turbine.kernel.wave.constraints import MMAType
from iree.turbine.kernel.wave.utils import (device_randn, device_zeros,
                                            get_default_run_config)
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw

torch.set_printoptions(linewidth=300)


# We want each row to contain [0 .. num_cols]
def reference_row(rows: int, cols: int):
    row_indices = torch.arange(cols).unsqueeze(0).expand(rows, cols)
    return row_indices


# We want each col to contain [0 .. num_rows]
def reference_col(rows: int, cols: int):
    col_indices = torch.arange(rows).unsqueeze(1).expand(rows, cols)
    return col_indices


def reference_row_plus_col(rows: int, cols: int):
    return reference_row(rows, cols) + reference_col(rows, cols)


# Input sizes
M = tkl.sym.M
N = tkl.sym.N
K = tkl.sym.K
# Workgroup tile sizes
ITERATIONS_OF_M_PER_WAVE = tkl.sym.ITERATIONS_OF_M_PER_WAVE
ITERATIONS_OF_N_PER_WAVE = tkl.sym.ITERATIONS_OF_N_PER_WAVE
BLOCK_K = tkl.sym.BLOCK_K
# Address space (for GPU, shared(1) or global(0))
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
# Other hyperparameters
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


# yapf: disable
def run_harness(fun: Callable, vM: int, vN: int, *args) -> bool:
    config = get_default_run_config()
    # Override manually to run.
    config = {"backend": "rocm", "device": "hip", "target": "gfx90a"}
    with TestLaunchContext({M: vM, N: vN},
        canonicalize=True,
        run=True,
        run_config=config):
        return fun(*args)


# yapf: disable
### Setting all of the following at the same time to agree on the same value works:
#   - ITERATIONS_OF_N_PER_WAVE
#   - VECTOR_SHAPE_N
#   - ELEMENTS_PER_THREAD_STORE
params = [
#   SIZE_M, SIZE_N, ITERATIONS_OF_M_PER_WAVE, ITERATIONS_OF_N_PER_WAVE, VECTOR_SHAPE_M, VECTOR_SHAPE_N, ELEMENTS_PER_THREAD_STORE
  [     4, 8, 1, 1, 1, 1, 1, ],
  [     4, 8, 1, 2, 1, 2, 2, ],
  [     4, 8, 1, 3, 1, 3, 3, ],
  [     4, 8, 1, 4, 1, 4, 4, ],
]
### However, The slightest discrepancy throws the TK compiler off:
# params = [
#   SIZE_M, SIZE_N, ITERATIONS_OF_M_PER_WAVE, ITERATIONS_OF_N_PER_WAVE, VECTOR_SHAPE_M, VECTOR_SHAPE_N, ELEMENTS_PER_THREAD_STORE
#   [     4, 8, 1, 1, 1, 4, 4, 4, ], # Tile size must be divisible by wave count and vector size, got: tile_size=1, wave_count=1, vector_size=4
#   [     4, 8, 1, 4, 1, 1, 4, 4, ], # MISCOMPILE INCORRECT RESULTS
#   [     4, 8, 1, 4, 1, 4, 1, 4, ], # CRASH TK COMPILER: Shape doesn't match: (1,) and (4,) in register cast_M:0_N:0 and elements_per_thread 4
#   [     4, 8, 1, 4, 1, 4, 4, 1, ], # CRASH TK COMPILER: Shape doesn't match: (4,) and (1,) in register cast_M:0_N:0 and elements_per_thread 1
# ]

for p in params:
    SIZE_M, \
    SIZE_N, \
    ITERATIONS_OF_M_PER_WAVE, \
    ITERATIONS_OF_N_PER_WAVE, \
    VECTOR_SHAPE_M, \
    VECTOR_SHAPE_N, \
    ELEMENTS_PER_THREAD_STORE = p

    workgroup_constraints = [
        [tkw.WorkgroupConstraint(M, ITERATIONS_OF_M_PER_WAVE, 0)],
        [tkw.WorkgroupConstraint(N, ITERATIONS_OF_N_PER_WAVE, 1)],
        [
            tkw.WorkgroupConstraint(M, ITERATIONS_OF_M_PER_WAVE, 0),
            tkw.WorkgroupConstraint(N, ITERATIONS_OF_N_PER_WAVE, 1)
        ],
    ]
    wave_constraints = [
        [],
        [tkw.WaveConstraint(M, 1)],
        [tkw.WaveConstraint(N, 1)],
        [tkw.WaveConstraint(M, 1), tkw.WaveConstraint(N, 1)],
        [tkw.WaveConstraint(M, 2)],
        [tkw.WaveConstraint(N, 2)],
        [tkw.WaveConstraint(M, 2), tkw.WaveConstraint(N, 2)],
        [tkw.WaveConstraint(M, 2), tkw.WaveConstraint(N, 2)],
        [tkw.WaveConstraint(M, 1), tkw.WaveConstraint(N, 2)],
        [tkw.WaveConstraint(M, 2), tkw.WaveConstraint(N, 1)],
    ]
    # yapf: enable

    for wgs in workgroup_constraints:
        for wvs in wave_constraints:
            unroll_N = True
            # In these stress tests compute self_index(N) and we want to distinguish
            # between the cases:
            #   1. there is a WorkgroupConstraint on N, therefore N is distributed
            #   and using ELEMENTS_PER_THREAD_INDEX == 1 results in proper
            #   propagations
            #   2. there is no WorkgroupConstraint on N, therefore N is unrolled and
            #   we have to use ELEMENTS_PER_THREAD_INDEX == ELEMENTS_PER_THREAD_STORE
            #   otherwise the TK compiler gets confused atm.
            # Ideally, in the future, things would just work out of the box without
            # having to adjust ELEMENTS_PER_THREAD_INDEX
            for wg in wgs:
                if wg.dim == N:
                    unroll_N = False

            # Skip this particular constraint if a WaveConstraint is set without
            # first setting the corresponding WorkgroupConstraint:
            # TK does not handle that case
            skip = False
            for wv in wvs:
                skip_wv = True
                for wg in wgs:
                    if wg.dim == wv.dim:
                        skip_wv = False
                if skip_wv:
                    skip = True
            if skip:
                continue

            ELEMENTS_PER_THREAD_INDEX = ELEMENTS_PER_THREAD_STORE if unroll_N else 1

            ###### User constraints
            constraints: list[tkw.Constraint] = []
            constraints += wgs
            constraints += wvs
            constraints += [
                tkw.HardwareConstraint(
                    threads_per_wave=64,
                    waves_per_block=(1, 1, 1),
                    vector_shapes={
                        M: VECTOR_SHAPE_M,
                        N: VECTOR_SHAPE_N
                    },
                )
            ]

            ###### Known cases to skip:
            # When we unroll N, TK does not handle imperfect unrolling (with a
            # remainder).
            if unroll_N and SIZE_N % ITERATIONS_OF_N_PER_WAVE != 0:
                continue

            @tkw.wave(constraints)
            def row(c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32]):
                i = tkw.self_index(
                    N, tkl.i64, elements_per_thread=ELEMENTS_PER_THREAD_INDEX)
                res = tkw.cast(i, tkl.f32)
                tkw.write(res,
                          c,
                          elements_per_thread=ELEMENTS_PER_THREAD_STORE)

            def fun_row(debug: bool = False) -> bool:
                c = device_zeros(SIZE_M, SIZE_N, dtype=torch.float32)
                if debug:
                    print(row(c).module_op)
                    return True
                else:
                    row(c)
                    correct = torch.all(
                        torch.isclose(reference_row(SIZE_M, SIZE_N),
                                      c.cpu().to(dtype=torch.int64))).item()
                    if not correct:
                        print(f"reference:\n{reference_row(SIZE_M, SIZE_N)}")
                        print(f"actual:\n{c.cpu().to(dtype=torch.int64)}")
                        print(
                            f"delta:\n{c.cpu().to(dtype=torch.int64) - reference_row(SIZE_M, SIZE_N)}"
                        )
                    return correct

            correct = run_harness(fun_row, SIZE_M, SIZE_N)
            if not correct:
                print(f"\nError under stress test constraints: {constraints}")
                run_harness(fun_row, SIZE_M, SIZE_N, True)
                assert correct, "Incorrect execution: ran in debug mode now stop"
