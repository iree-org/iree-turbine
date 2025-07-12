# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import functools
from typing import Callable, Any
from ..compile_options import WaveCompileOptions
from itertools import chain
from iree.turbine.kernel.lang import IndexSymbol


@functools.lru_cache
def compute_grid(kernel_dynamic_dims: tuple[int], grid_fn: Callable):
    return [int(x) for x in grid_fn(list(kernel_dynamic_dims))]


def write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


def print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        write_file(filename, "w", res)
    else:
        print(res)


def invoke_with_wave_runtime(
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    scalar_args: list[int | float],
    bound_scalar_symbols: dict[IndexSymbol, int],
    dynamic_symbols: list[int],
    gpu_func: Any,
):
    """
    Invokes the kernel with the wave runtime.
    """
    import wave_runtime

    num_inputs = len(kernel_inputs)
    dynamic_dims = tuple(
        scalar_args[v - num_inputs] for v in bound_scalar_symbols.values()
    ) + tuple(dynamic_symbols)
    # Update the grid size as this may vary depending
    # on the dynamic symbols.
    grid = compute_grid(dynamic_dims, options.kernel_launch_info.grid)

    # Populate all the information required to launch the kernel.
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        gpu_func,
        options.kernel_launch_info.shared_memory_bytes,
        grid[0],
        grid[1],
        grid[2],
        options.kernel_launch_info.blocks[0],
        options.kernel_launch_info.blocks[1],
        options.kernel_launch_info.blocks[2],
    )

    # Ensure that the tensors are contiguous.
    kern_args = []
    for arg_tensor in chain(kernel_inputs, kernel_outputs):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        kern_args.append(arg_tensor.data_ptr())

    kernel_args = wave_runtime.Int64Vector(kern_args)
    dyn_dims = wave_runtime.Int64Vector(dynamic_dims[len(bound_scalar_symbols) :])
    # Launch the kernel.
    wave_runtime.launch(kernel_launch_info, kernel_args, dyn_dims, scalar_args)


def get_default_arch() -> str:
    """Return default ROCM architecture"""
    if not torch.cuda.is_available():
        return "cpu"
    device = torch.device("cuda")
    gcnArch = torch.cuda.get_device_properties(device).gcnArchName
    assert "gfx" in gcnArch, "Currently only support GFX/ROCm for get_default_arch."
    # The gcnArchName comes back like gfx90a:sramecc+:xnack.
    colon_pos = gcnArch.find(":")
    return gcnArch[0:colon_pos]


def set_default_run_config(options: WaveCompileOptions) -> WaveCompileOptions:
    """Return default config for running."""
    options.backend = "rocm"
    options.target = get_default_arch()
    return options
