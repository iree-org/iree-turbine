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

# Symbols
TOKENS_IN_CHUNK, E, M, N, K, TOPK = (
    tkl.sym.TOKENS_IN_CHUNK,
    tkl.sym.E,
    tkl.sym.M,
    tkl.sym.N,
    tkl.sym.K,
    tkl.sym.TOPK,
)

BLOCK_TOKENS_IN_CHUNK, BLOCK_E, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_TOPK = (
    tkl.sym.BLOCK_TOKENS_IN_CHUNK,
    tkl.sym.BLOCK_E,
    tkl.sym.BLOCK_M,
    tkl.sym.BLOCK_N,
    tkl.sym.BLOCK_K,
    tkl.sym.BLOCK_TOPK,
)

LOAD_TOKS_PER_THREAD, LOAD_ELEMS_PER_THREAD, STORE_ELEMS_PER_THREAD = (
    tkl.sym.LOAD_TOKS_PER_THREAD,
    tkl.sym.LOAD_ELEMS_PER_THREAD,
    tkl.sym.STORE_ELEMS_PER_THREAD,
)

ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE


def harness(build_constraints_fun, kernel_fun, *args, **kwargs):
    constraints = build_constraints_fun(*args, **kwargs)
    with tk.gen.TestLaunchContext(
        kwargs["static_symbols"] if "static_symbols" in kwargs else {}
    ):
        lw = LaunchableWave(constraints, "kernel_fun", kernel_fun)

        trace: CapturedTrace = lw._trace()
        idxc: IndexingContext = IndexingContext.current()
        graph_passes = lw.build_initial_pass_pipeline(trace, print_ir_before=["all"])
        for p in graph_passes:
            try_apply_pass(p, trace, print_ir_before=["all"])

        lw.infer_grid_shape(idxc)

        compile_config = get_default_compile_config()
        with Context() as context:
            mb, trace, exe, kernel_sig, entrypoint_name = lw.compile_to_mlir(
                trace, context, **kwargs, compile_config=compile_config
            )
            print(mb.module_op)


def build_block_constraints(*args, **kwargs) -> Sequence[tkw.Constraint]:
    constraints: list[tkw.Constraint] = []
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]
    # constraints += [tkw.WorkgroupConstraint(TOKENS_IN_CHUNK, BLOCK_TOKENS_IN_CHUNK, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]

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
            TOKENS_IN_CHUNK: 33,
            E: 2,
            M: 16,
            N: 16,
            K: 32,
            TOPK: 33,
            BLOCK_M: 16,
            BLOCK_N: 16,
            BLOCK_K: 16,
            # BLOCK_TOKENS_IN_CHUNK: 16,
            LOAD_TOKS_PER_THREAD: 1,
            LOAD_ELEMS_PER_THREAD: 4,
            STORE_ELEMS_PER_THREAD: 1,
        },
        "vector_shapes": {
            TOKENS_IN_CHUNK: 1,
            E: 1,
            M: 16,
            N: 16,
            K: 16,
            TOPK: 0,
        },
        "canonicalize": {True},
    }


i, j, k = [tkw.IndexMapping.iterator(i) for i in range(3)]
d0 = tkw.IndexMapping.dynamic_val(0)

offset_mapping_a = tkw.IndexMapping(
    num_iterators=2,
    inputs={TOKENS_IN_CHUNK: d0, K: j},
    outputs={TOKENS_IN_CHUNK: i, K: j},
    dynamic_val_mappings=({TOKENS_IN_CHUNK: i}),
)

offset_mapping_b = tkw.IndexMapping(
    num_iterators=3,
    inputs={E: d0, N: j, K: k},
    outputs={E: i, N: j, K: k},
    dynamic_val_mappings=({E: i}),
)


def fused_moe_kernel(
    A: tkl.Memory[TOKENS_IN_CHUNK, K, ADDRESS_SPACE, tkl.f16],
    B: tkl.Memory[E, N, K, ADDRESS_SPACE, tkl.f16],
    C: tkl.Memory[M, TOPK, N, ADDRESS_SPACE, tkl.f32],
    # Quantization parameters.
    # a_scale_ptr,
    # b_scale_ptr,
    # Useful only if MUL_ROUTED_WEIGHT is True
    # topk_weights_ptr,
    sorted_token_ids: tkl.Memory[M, ADDRESS_SPACE, tkl.i64],
    expert_ids: tkl.Memory[M, ADDRESS_SPACE, tkl.i64],
    # Unclear what to do with this ..
    # num_tokens_post_padded_ptr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    # num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    # if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
    #     return

    c_reg = tkl.Register[TOKENS_IN_CHUNK, K, E, N, M, TOPK, tkl.f32](0.0)

    tok_id = tkw.read(sorted_token_ids, elements_per_thread=LOAD_TOKS_PER_THREAD)
    expert_id = tkw.read(expert_ids, elements_per_thread=LOAD_TOKS_PER_THREAD)

    @tkw.reduction(K, init_args=[c_reg])
    def repeat(
        acc: tkl.Register[TOKENS_IN_CHUNK, K, E, N, M, TOPK, tkl.f32]
    ) -> tkl.Register[TOKENS_IN_CHUNK, K, E, N, M, TOPK, tkl.f32]:
        a_reg = tkw.read(
            A,
            mapping=offset_mapping_a,
            mapping_dynamic_vals=(tok_id,),
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )

        b_reg = tkw.read(
            B,
            mapping=offset_mapping_b,
            mapping_dynamic_vals=(expert_id,),
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
        )

        acc = tkw.mma(a_reg, b_reg, acc)
        return acc

    tkw.write(repeat, C, elements_per_thread=STORE_ELEMS_PER_THREAD)


if __name__ == "__main__":

    @run_test
    def static_correct_1():
        cfg = copy.deepcopy(config())
        harness(build_block_constraints, fused_moe_kernel, **cfg)
