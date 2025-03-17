# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import ctypes

from typing import Optional

import iree.runtime as rt
import torch

from .manager import KernelCacheEntry
from ..compiler.kernel_codegen import KernelBufferUsage
from .._support.indexing import IndexExpr
from .._support.profiling import benchmark_module

__all__ = [
    "RUNTIME_CACHE",
    "get_device_uuid",
    "invoke_cached_kernel",
    "invoke_vmfb",
]

# Cache for the system context and vm function.
RUNTIME_CACHE: dict[str, tuple[rt.SystemContext, rt.VmFunction]] = {}


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
        capsule = arg_tensor.__dlpack__(None)
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
    for dynamic_dim in dynamic_dims:
        if isinstance(dynamic_dim, int):
            arg_list.push_int(dynamic_dim)
        else:
            raise ValueError(f"Unsupported dynamic dim type: {type(dynamic_dim)}")

    vm_context.invoke(entry_function, arg_list, ret_list)


def _write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


def _print_bench_result(result, filename):
    import json

    res = json.dumps(result, sort_keys=True, indent=4)
    if filename is not None:
        _write_file(filename, "w", res)
    else:
        print(res)


def get_device_uuid(input_tensors: list[torch.Tensor]) -> tuple[int, str]:
    """
    Checks all torch.Tensor are on the same device, and get UUID from Torch device.
    """
    device_list = [
        input.device for input in input_tensors if isinstance(input, torch.Tensor)
    ]
    if len(set(device_list)) != 1:
        raise ValueError(f"Found multiple device on input tensors:{set(device_list)}")
    device = device_list[0]
    if device.type != "cuda":
        raise ValueError("Expected all argument tensors to be in GPU.")
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return uuid


def invoke_cached_kernel(
    cached_kernel: KernelCacheEntry,
    args: list[torch.Tensor],
    config: dict[str, str],
    dynamic_symbols: list[IndexExpr],
    dynamic_symbols_map: dict[IndexExpr, int],
    run: bool,
    run_bench: bool,
    inplace: bool = True,
):
    if not config:
        raise ValueError("no config provided")

    kernel_inputs = []
    kernel_outputs = []
    for arg, usage in zip(args, cached_kernel.kernel_sig):
        if usage == KernelBufferUsage.INPUT:
            kernel_inputs.append(arg)

        if usage == KernelBufferUsage.OUTPUT:
            kernel_outputs.append(arg)

    kernel_dynamic_dims = []
    if dynamic_symbols:
        kernel_dynamic_dims = dynamic_symbols_map.values()

    invoke_vmfb(
        cached_kernel.vmfb,
        cached_kernel.function_name,
        config,
        kernel_inputs,
        kernel_outputs,
        kernel_dynamic_dims,
        run,
        run_bench,
        inplace=inplace,
        kernel_hash=cached_kernel.cache_id,
    )


def invoke_vmfb(
    vmfb: bytes,
    func_name: str,
    config: dict[str, str],
    kernel_inputs: list[torch.Tensor],
    kernel_outputs: list[torch.Tensor],
    kernel_dynamic_dims: list[int] = [],
    run: bool = False,
    run_bench: bool = False,
    inplace: bool = False,
    kernel_hash: Optional[str] = None,
):

    device = config["device"]
    if run_bench:
        bench_batch_size = config.get("benchmark_batch_size", None)
        bench_repetitions = config.get("benchmark_repetitions", None)
        bench_file = config.get("benchmark_results_file", None)

        benchmark_flags = {}

        # If we use 1000 for bench_batch_size during compilation, and set this batch size to 1,
        # then the latency is in milliseconds.
        benchmark_flags["batch_size"] = 1

        if bench_repetitions is not None:
            benchmark_flags["benchmark_repetitions"] = int(bench_repetitions)

    if not (run or run_bench):
        return

    if inplace and not device.startswith("local"):
        # Select device as the GPU, where input tensors are coming from.
        device_uuid = get_device_uuid(kernel_inputs + kernel_outputs)
        device = f"{device}://GPU-{device_uuid}"
    rt_config = rt.Config(device)
    device = rt_config.device
    vm_instance = rt_config.vm_instance

    if kernel_hash and kernel_hash in RUNTIME_CACHE:
        ctx, func = RUNTIME_CACHE[kernel_hash]
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
        func = mod.lookup_function(func_name)
        if kernel_hash:
            RUNTIME_CACHE[kernel_hash] = (ctx, func)

    if run:
        if inplace:
            _inplace_invoke(
                ctx.vm_context,
                device,
                func,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
            )
        else:
            _invoke(
                ctx.vm_context,
                device,
                func,
                kernel_inputs,
                kernel_outputs,
                kernel_dynamic_dims,
            )

    if run_bench:
        benchmark_results = benchmark_module(
            kernel_inputs,
            kernel_outputs,
            kernel_dynamic_dims,
            config,
            inplace,
            mod,
            entry_function=func_name,
            device=device,
            **benchmark_flags,
        )
        _print_bench_result(benchmark_results, bench_file)
