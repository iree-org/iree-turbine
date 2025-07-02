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
from ..profiling import benchmark_module
from itertools import chain
from warnings import warn
from iree.turbine.kernel.lang import IndexSymbol

# Cache for the system context and vm function.
RUNTIME_CACHE: dict[str, tuple[rt.SystemContext, rt.VmFunction]] = {}


@functools.lru_cache
def compute_grid(kernel_dynamic_dims: tuple[int], grid_fn: Callable):
    return [int(x) for x in grid_fn(list(kernel_dynamic_dims))]


def _write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


@functools.lru_cache
def get_device_uuid(device_list: list[str], device_str: str) -> tuple[int, str]:
    """
    Checks all torch.Tensor are on the same device, and get UUID from Torch device.
    """
    if len(set(device_list)) != 1:
        raise ValueError(f"Found multiple device on input tensors:{set(device_list)}")
    device = device_list[0]
    if device.type != "cuda":
        raise ValueError("Expected all argument tensors to be in GPU.")
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    device_str = f"{device_str}://GPU-{uuid}"
    return device_str


def _inplace_invoke(
    vm_context, device, entry_function, inputs, outputs, scalar_args, dynamic_dims
):
    linearized_arg_len = (
        len(inputs) + len(outputs) + len(scalar_args) + len(dynamic_dims)
    )
    # ret_list is 0 because we modify/write result in place.
    arg_list = rt.VmVariantList(linearized_arg_len)
    ret_list = rt.VmVariantList(0)

    def push_tensor_to_arg_list(arg_tensor: torch.Tensor):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        capsule = torch.to_dlpack(arg_tensor)
        arg_tensor_bv = device.from_dlpack_capsule(capsule)
        arg_list.push_ref(arg_tensor_bv)

    # Linearize arguments, In linearized arg_list, we first push in all inputs,
    # then all the outputs, and lastly all the dynamic dims.
    for input in chain(inputs, outputs, scalar_args, dynamic_dims):
        if isinstance(input, torch.Tensor):
            push_tensor_to_arg_list(input)
        elif isinstance(input, int):
            arg_list.push_int(input)
        elif isinstance(input, float):
            warn("Currently, `push_float` is not working on the iree side")
            arg_list.push_float(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    try:
        vm_context.invoke(entry_function, arg_list, ret_list)
    except ValueError as e:
        raise RuntimeError(
            f"Error invoking IREE\n{entry_function}\n"
            f"{arg_list=}\n"
            f"inputs: {', '.join([str(i.shape) for i in inputs])}\n"
            f"outputs: {', '.join([str(o.shape) for o in outputs])}"
        ) from e


def _print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        _write_file(filename, "w", res)
    else:
        print(res)


def invoke_vmfb(
    vmfb: bytes,
    options: WaveCompileOptions,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    scalar_args: list[int | float] = [],
    bound_scalar_symbols: dict[IndexSymbol, int] = {},
    dynamic_symbols: list[int] = [],
    gpu_func: Optional[Any] = None,
):
    if options.wave_runtime:
        invoke_with_wave_runtime(
            options,
            kernel_inputs,
            kernel_outputs,
            scalar_args,
            bound_scalar_symbols,
            dynamic_symbols,
            gpu_func,
        )
        return

    device = options.device
    if options.run_bench:
        benchmark_flags = {}
        # If we use 1000 for bench_batch_size during compilation, and set this batch size to 1,
        # then the latency is in milliseconds.
        benchmark_flags["batch_size"] = 1

        if options.benchmark_repetitions is not None:
            benchmark_flags["benchmark_repetitions"] = int(
                options.benchmark_repetitions
            )

    # Select device as the GPU, where input tensors are coming from.
    device_list = tuple(
        input.device
        for input in kernel_inputs + kernel_outputs
        if isinstance(input, torch.Tensor)
    )
    device = get_device_uuid(device_list, device)

    rt_config = rt.Config(device)
    device = rt_config.device
    vm_instance = rt_config.vm_instance

    if options.kernel_hash and options.kernel_hash in RUNTIME_CACHE:
        ctx, func = RUNTIME_CACHE[options.kernel_hash]
    else:
        mod = rt.VmModule.copy_buffer(vm_instance, vmfb)
        vm_modules = [
            mod,
            rt.create_hal_module(vm_instance, device),
        ]
        ctx = rt.SystemContext(
            vm_modules=vm_modules,
            config=rt_config,
        )
        func = mod.lookup_function(options.func_name)
        if options.kernel_hash:
            RUNTIME_CACHE[options.kernel_hash] = (ctx, func)

    _inplace_invoke(
        ctx.vm_context,
        device,
        func,
        kernel_inputs,
        kernel_outputs,
        scalar_args,
        dynamic_symbols,
    )

    if options.run_bench:
        benchmark_results = benchmark_module(
            options,
            kernel_inputs,
            kernel_outputs,
            vmfb,
            options.func_name,
            **benchmark_flags,
        )
        _print_bench_result(benchmark_results, options.bench_file)


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
    options.device = "hip"
    options.target = get_default_arch()
    return options
