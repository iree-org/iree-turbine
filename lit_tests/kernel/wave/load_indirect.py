# RUN: python %s | FileCheck %s

import copy
import logging
from typing import Sequence

import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.wave.expansion.expansion import expand_graph
from iree.turbine.kernel.wave.index_sequence_analysis import (
    set_node_indices,
    set_post_expansion_indices,
)
from iree.turbine.kernel.compiler.ir import Context, Location, Module
from iree.turbine.kernel.wave.type_inference import infer_types
from iree.turbine.kernel.wave.wave import LaunchableWave
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel._support.indexing import IndexingContext
from iree.turbine.kernel._support.tracing import CapturedTrace
from iree.turbine.kernel.wave.utils import (
    get_default_compile_config,
    print_trace,
    run_test,
    try_apply_pass,
)
from moe import *


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            # One must always specify mma_type or vector_shapes.
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes=kwargs["vector_shapes"] if "vector_shapes" in kwargs else {},
        )
    ]
    return constraints


def config():
    return {
        "static_symbols": {
            M: 16,
            N: 16,
            K: 32,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            LOAD_TOKS_PER_THREAD: 4,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 4,
        },
        "canonicalize": {True},
    }


m, n = [tkw.IndexMapping.iterator(i) for i in range(2)]
d0 = tkw.IndexMapping.dynamic_val(0)

offset_mapping_a = tkw.IndexMapping(
    num_iterators=2,
    inputs={M: d0, N: n},
    outputs={M: m, N: n},
    dynamic_val_mappings=({M: m}),
)


def load_indirect(
    A: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    B: tkl.Memory[M, ADDRESS_SPACE, tkl.i64],
    C: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f32],
):
    idx_i64 = tkw.read(B, elements_per_thread=LOAD_TOKS_PER_THREAD)
    idx = tkw.cast(idx_i64, tkl.index)
    a_reg = tkw.read(
        A,
        mapping=offset_mapping_a,
        mapping_dynamic_vals=(idx,),
        elements_per_thread=LOAD_ELEMS_PER_THREAD,
    )

    tkw.write(a_reg, C, elements_per_thread=STORE_ELEMS_PER_THREAD)


if __name__ == "__main__":

    @run_test
    def load_indirect():
        cfg = copy.deepcopy(config())
        cfg["vector_shapes"] = {M: 16, N: 16}
        harness(build_block_constraints, load_indirect, **cfg)
