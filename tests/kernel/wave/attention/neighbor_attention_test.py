import pytest
import torch
import iree.turbine.kernel as tk
import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.torch_utils import (
    device_zeros,
    device_arange,
    device_randint,
    device_ones,
)
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType, MMAOperand, GenericDot

import os
from torch.nn import functional as F
from ..common.utils import (
    require_e2e,
    require_cdna3,
    dump_generated_mlir,
    perf_test,
    param_bool,
    enable_scheduling_barriers,
)
from ..common.shapes import get_test_shapes
from torch.testing import assert_close

from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions


K1 = tkl.sym.K1
M = tkl.sym.M
N = tkl.sym.N
E = tkl.sym.E
FT = tkl.sym.FT
FT2 = tkl.sym.FT2
F_IN = tkl.sym.F_IN
F_OUT = tkl.sym.F_OUT
BLOCK_M = tkl.sym.BLOCK_M
BLOCK_N = tkl.sym.BLOCK_N
BLOCK_K1 = tkl.sym.BLOCK_K1
ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD


@require_e2e
@pytest.mark.parametrize(
    "mfma_variant",
    [
        GenericDot(k_mult=32, k_vec_size=1, out_vec_size=1, along_dim=MMAOperand.M),
    ],
)
def test_neighbor_attention(mfma_variant: MMAType):

    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WaveConstraint(M, BLOCK_M / 1),
        tkw.WaveConstraint(N, BLOCK_N),
        tkw.TilingConstraint(K1, BLOCK_K1),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(1, 1, 1),
            mma_type=mfma_variant,
            vector_shapes={M: 2, N: 1, K1: 32},
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    d0 = tkw.IndexMapping.dynamic_val(0)
    mapping_scatter = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    scale = 1.0

    @tkw.wave(constraints)
    def neighbor_attention(
        concat_dst_edge_features: tkl.Memory[M, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        mlp_weights: tkl.Memory[N, K1, GLOBAL_ADDRESS_SPACE, tkl.f32],
        edge_dest: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.i32],
        out_V: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        edge_scaling = tkl.Register[M, N, tkl.f32](scale)
        zero_accumulator = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K1, init_args=[zero_accumulator])
        def accumulate_dot_product(
            partial_sum: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            # Gather destination node features, pass through MLP and obtain attention score for each edge : perform h_V_dst * mlp_weight
            concat_feat_reg = tkw.read(
                concat_dst_edge_features, elements_per_thread=LOAD_ELEMS_PER_THREAD
            )
            mlp_reg = tkw.read(mlp_weights, elements_per_thread=LOAD_ELEMS_PER_THREAD)
            partial_sum = tkw.mma(concat_feat_reg, mlp_reg, partial_sum)
            return partial_sum

        # Normalize attention scores
        scaled_scores = accumulate_dot_product * edge_scaling
        substract = scaled_scores - zero_accumulator

        edge_dest_reg = tkw.read(edge_dest)

        tkw.write(
            substract,
            out_V,
            elements_per_thread=LOAD_ELEMS_PER_THREAD,
            mapping=mapping_scatter,
            mapping_dynamic_vals=(edge_dest_reg,),
        )


    # Hyperparams
    hyperparams = {
        ADDRESS_SPACE: tkl.AddressSpace.SHARED_MEMORY.value,
        M: 32,
        N: 1,
        K1: 32,
        BLOCK_M: 32,
        BLOCK_N: 1,
        BLOCK_K1: 32,
        LOAD_ELEMS_PER_THREAD: 1,
        STORE_ELEMS_PER_THREAD: 1,
    }

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        run_bench=False,
        schedule=SchedulingType.NONE,
        use_scheduling_barriers=False,
        compile_to_mlir=False,
        print_signature=True,
    )
    options = set_default_run_config(options)
    neighbor_attention = wave_compile(options, neighbor_attention)
    print(neighbor_attention.asm)

    concat_dst_edge_features = (
        device_arange(32 * 32, dtype=torch.float32).reshape(32, 32).contiguous()
    )
    mlp_weight = device_ones(32, dtype=torch.float32).reshape(1, 32).contiguous()
    edge_dest = device_randint(0, 10, (32, 1), dtype=torch.int32).contiguous()

    output = device_zeros(32, dtype=torch.float32).reshape(32, 1).contiguous()
    neighbor_attention(
        concat_dst_edge_features,
        mlp_weight,
        edge_dest,
        output,
    )