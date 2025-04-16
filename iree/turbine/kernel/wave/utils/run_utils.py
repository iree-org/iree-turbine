# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
import torch
import functools
import iree.runtime as rt
from typing import Callable
import ctypes
import glob
from ..compile_options import WaveCompileOptions
from .compile_utils import compile_to_vmfb
from .classes import KernelLaunchInfo
from ..profiling import benchmark_module


# Cache for the system context and vm function.
RUNTIME_CACHE: dict[str, tuple[rt.SystemContext, rt.VmFunction]] = {}


@functools.lru_cache
def compute_grid(kernel_dynamic_dims: tuple[int], grid_fn: Callable):
    return [int(x) for x in grid_fn(list(kernel_dynamic_dims))]


def _read_file(name, mode):
    with open(name, mode) as file:
        data = file.read()
    return data


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


_dl_tensor_name = ctypes.create_string_buffer(b"dltensor")
_set_capsule_name = ctypes.pythonapi.PyCapsule_SetName


def _inplace_invoke(vm_context, device, entry_function, inputs, outputs, dynamic_dims):
    linearized_arg_len = len(inputs) + len(outputs) + len(dynamic_dims)
    # ret_list is 0 because we modify/write result in place.
    arg_list = rt.VmVariantList(linearized_arg_len)
    ret_list = rt.VmVariantList(0)

    def push_tensor_to_arg_list(arg_tensor: torch.Tensor):
        if not arg_tensor.is_contiguous():
            arg_tensor = arg_tensor.contiguous()
        capsule = torch.to_dlpack(arg_tensor)
        arg_tensor_bv = device.from_dlpack_capsule(capsule)

        # IREE runtime renames capsule to "dltensor_used" for some reason, but
        # only deletes capsules with "dltensor" name, which is causing a memory
        # leak.
        _set_capsule_name(ctypes.py_object(capsule), _dl_tensor_name)
        arg_list.push_ref(arg_tensor_bv)

    # Linearize arguments, In linearized arg_list, we first push in all inputs,
    # then all the outputs, and lastly all the dynamic dims.
    for input in inputs:
        if isinstance(input, torch.Tensor):
            push_tensor_to_arg_list(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
    for output in outputs:
        if isinstance(output, torch.Tensor):
            push_tensor_to_arg_list(output)
        else:
            raise ValueError(f"Unsupported output type: {type(output)}")
    # we want scalars to be at the end during codegen/dispatch to iree
    # to maintain the consistency.
    for input in inputs:
        if isinstance(input, (float, int)):
            # arg_list.push_float(input)
            # Currently, `push_float` is not working on the iree side.
            raise NotImplementedError("Float inputs are not supported.")

    for dynamic_dim in dynamic_dims:
        if isinstance(dynamic_dim, int):
            arg_list.push_int(dynamic_dim)
        else:
            raise ValueError(f"Unsupported dynamic dim type: {type(dynamic_dim)}")

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
):
    if options.wave_runtime:
        invoke_with_wave_runtime(options, kernel_inputs, kernel_outputs)
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

    if options.inplace:
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

    if options.inplace:
        _inplace_invoke(
            ctx.vm_context,
            device,
            func,
            kernel_inputs,
            kernel_outputs,
            options.dynamic_symbols_map.values(),
        )
    else:
        _invoke(
            ctx.vm_context,
            device,
            func,
            kernel_inputs,
            kernel_outputs,
            options.dynamic_symbols_map.values(),
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
):
    """
    Invokes the kernel with the wave runtime.
    """
    import wave_runtime
    from ..cache import WAVE_RUNTIME_DIR, CACHE_BASE_DIR

    # Get the path to the binary.
    if options.kernel_hash:
        binary = (
            str(CACHE_BASE_DIR / options.kernel_hash / options.kernel_hash) + ".hsaco"
        )
    else:
        binary = glob.glob(str(WAVE_RUNTIME_DIR / "*.hsaco"))[0]

    dynamic_dims = tuple(options.dynamic_symbols_map.values())
    # Update the grid size as this may vary depending
    # on the dynamic symbols.
    grid = compute_grid(dynamic_dims, options.kernel_launch_info.grid)

    # Populate all the information required to launch the kernel.
    hash_str = "" if not options.kernel_hash else options.kernel_hash
    kernel_launch_info = wave_runtime.KernelLaunchInfo(
        binary,
        options.kernel_launch_info.func_name,
        hash_str,
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


def compile_and_invoke(
    asm: str,
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    options: WaveCompileOptions,
):
    compiled_wave_vmfb = compile_to_vmfb(asm, options)
    invoke_vmfb(
        compiled_wave_vmfb,
        options,
        kernel_inputs,
        kernel_outputs,
    )


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
