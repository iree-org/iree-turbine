# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch import nn
from torch.testing import assert_close
import math

import iree.turbine.kernel.lang as tkl
import iree.turbine.kernel.wave as tkw
from iree.turbine.kernel.lang.global_symbols import *
from iree.turbine.kernel.wave.iree_utils import generate_iree_ref
from iree.turbine.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from iree.turbine.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.turbine.kernel.wave.utils.mma_utils import (
    get_mfma_load_elems_per_thread,
    get_mfma_store_elems_per_thread,
)
from iree.turbine.kernel.wave.scheduling.schedule import SchedulingType
from iree.turbine.kernel.wave.compile import WaveCompileOptions, wave_compile
from iree.turbine.kernel.wave.constraints import MMAType


def get_linear_kernel(
    shape: tuple[int],
    dynamic_dims: bool = False,
    mfma_variant: MMAType = MMAType.F32_16x16x16_F16,
    use_bias: bool = False,
):
    # Input sizes
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    # Workgroup tile sizes
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    # Address space (for GPU, shared(1) or global(0))
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    # Other hyperparameters
    LOAD_ELEMS_PER_THREAD = tkl.sym.LOAD_ELEMS_PER_THREAD
    STORE_ELEMS_PER_THREAD = tkl.sym.STORE_ELEMS_PER_THREAD

    # Expose user-constraints
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 2)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            waves_per_block=(2, 2, 1),
            vector_shapes={B: 0},
            mma_type=mfma_variant,
        )
    ]

    # With dynamic dimensions, we need to add an assumption on how big
    # the iterate dimension is to determine whether we can schedule or not.
    if dynamic_dims:
        constraints += [tkw.Assumption(K > BLOCK_K * 4)]

    # Wave-level micro-kernel.
    # Since warps are not directly addressable, there is no
    # explicit notion of a warp id (like a workgroup or thread id).
    # This kernel uses the input sizes M, N, K throughout, as the tiling
    # and data movement strategy is determined during the compilation process.
    # These can be influenced by introducing constraints.
    def gemm_core(a, b, c_reg, result):
        @tkw.iterate(K, init_args=[c_reg])
        def repeat(
            acc: tkl.Register[B, M, N, tkl.f32],
        ) -> tkl.Register[B, M, N, tkl.f32]:
            a_reg = tkw.read(a)
            b_reg = tkw.read(b)
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(
            tkw.cast(repeat, tkl.f16),
            result,
        )

    @tkw.wave(constraints)
    def gemm(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        result: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        c_reg = tkl.Register[B, M, N, tkl.f32](0.0)
        gemm_core(a, b, c_reg, result)

    @tkw.wave(constraints)
    def gemm_with_bias(
        a: tkl.Memory[B, M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        bias: tkl.Memory[N, ADDRESS_SPACE, tkl.f16],
        result: tkl.Memory[B, M, N, GLOBAL_ADDRESS_SPACE, tkl.f16],
    ):
        bias_reg = tkw.read(bias)
        bias_reg = tkw.broadcast(bias_reg, target_shape=[B, M, N])
        bias_reg = tkw.cast(bias_reg, tkl.f32)
        # We can get "free" bias-add by setting bias as the initial
        # value of accumulator to the mma
        gemm_core(a, b, bias_reg, result)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        LOAD_ELEMS_PER_THREAD: get_mfma_load_elems_per_thread(mfma_variant),
        STORE_ELEMS_PER_THREAD: get_mfma_store_elems_per_thread(mfma_variant),
        BLOCK_B: 1,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        N: shape[1],
        K: shape[0],
    }
    hyperparams.update(get_default_scheduling_params())

    dynamic_symbols = [B, M]

    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        dynamic_symbols=dynamic_symbols,
        dynamic_symbols_map={},
    )
    options = set_default_run_config(options)
    gemm_kernel = gemm
    if use_bias:
        gemm_kernel = gemm_with_bias
    compiled_gemm = wave_compile(options, gemm_kernel)
    return compiled_gemm


LINEAR_SUPPORTED_DTYPE = {torch.float16}


class WaveLinear(nn.Module):
    """Fork of nn.Linear implementation but modified to handle Wave Kernel"""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        device = device or torch.device("cuda:0")
        dtype = dtype or torch.float16

        if device.type != "cuda":
            raise ValueError(f"{self.__class__.__name__} only support GPU device.")
        if dtype not in LINEAR_SUPPORTED_DTYPE:
            raise ValueError(
                f"{self.__class__.__name__} does not support dtype: {dtype}."
            )
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # Wave related initialization
        self.kernel = get_linear_kernel([in_features, out_features], use_bias=bias)

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) >= 2
        # Determine parameter shapes
        input_len = input.shape[-2]
        batch = input.shape[0:-2]

        # Compute "flattened" batch shapes
        flat_batch = math.prod(batch)
        out_features = self.weight.shape[0]
        output_shape = [flat_batch, input_len, out_features]

        # Setup and run kernel
        output = torch.empty(
            output_shape, dtype=self.weight.dtype, device=self.weight.device
        )
        self.kernel.options.dynamic_symbols_map = {
            tkl.sym.B: flat_batch,
            tkl.sym.M: input_len,
        }
        if self.bias is None:
            self.kernel(
                input.view(flat_batch, input_len, input.shape[-1]), self.weight, output
            )
        else:
            self.kernel(
                input.view(flat_batch, input_len, input.shape[-1]),
                self.weight,
                self.bias,
                output,
            )

        # Return non flattened shape
        return output.view(*batch, input_len, out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
