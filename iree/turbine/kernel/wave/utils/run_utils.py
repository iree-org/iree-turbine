# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import torch
import functools
import iree.runtime as rt
from typing import Callable, Optional, Any
import ctypes
from ..compile_options import WaveCompileOptions
from .classes import KernelLaunchInfo
from ..profiling import benchmark_module


@functools.lru_cache
def compute_grid(kernel_dynamic_dims: tuple[int], grid_fn: Callable):
    return [int(x) for x in grid_fn(list(kernel_dynamic_dims))]


def _write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


def _invoke(vm_context, device, entry_function, inputs, outputs, dynamic_dims):
    arg_list = rt.VmVariantList(len(inputs) + len(dynamic_dims))
    ret_list = rt.VmVariantList(len(outputs))

    for input in inputs:
        if isinstance(input, torch.Tensor):
            input_cpu = input.cpu().contiguous()
            device_array = rt.asdevicearray(device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    for dynamic_dim in dynamic_dims:
        if isinstance(dynamic_dim, int):
            arg_list.push_int(dynamic_dim)
        else:
            raise ValueError(f"Unsupported dynamic dim type: {type(dynamic_dim)}")

    vm_context.invoke(entry_function, arg_list, ret_list)

    for i, ret in enumerate(outputs):
        device_buffer_view = rt.HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
        device_array = rt.DeviceArray(device, device_buffer_view)

        # TODO: Make to_host accept out array/buffer, so we can avoid extra data copy.
        host_array = device_array.to_host()

        # Convert to torch tensor without actually importing torch.
        ret[:] = type(ret)(host_array)


def print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        _write_file(filename, "w", res)
    else:
        print(res)


def invoke_with_wave_runtime(
    gpu_func: Any,
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
):
    """
    Invokes the kernel with the wave runtime.
    """
    import wave_runtime

    dynamic_dims = tuple(options.dynamic_symbols_map.values())
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
    scalar_args = []
    for arg_tensor in kernel_inputs + kernel_outputs:
        if isinstance(arg_tensor, (float, int)):
            scalar_args.append(arg_tensor)
            continue

        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()

        kern_args.append(arg_tensor.data_ptr())

    kernel_args = wave_runtime.Int64Vector(kern_args)
    dyn_dims = wave_runtime.Int64Vector(dynamic_dims)
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
    options.device = "hip"
    options.target = get_default_arch()
    return options
