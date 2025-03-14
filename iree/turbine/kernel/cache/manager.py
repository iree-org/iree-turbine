# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import json
import os
import shutil
import torch
import threading

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .compiler.kernel_codegen import KernelBufferUsage
from ._support.indexing import IndexExpr

default_cache_base_dir = Path.home() / ".cache" / "turbine_kernels"
CACHE_BASE_DIR = Path(os.environ.get("CACHE_DIR", default_cache_base_dir))
ALWAYS_COMPILE = int(os.environ.get("ALWAYS_COMPILE", 0))
CACHE_ON = int(os.environ.get("CACHE_ON", 1))
CACHE_LIMIT = int(os.environ.get("CACHE_LIMIT", 16))
MAX_LRU_CACHE_SIZE = int(os.environ.get("MAX_LRU_CACHE_SIZE", 128))


def is_cache_enabled() -> bool:
    return bool(CACHE_ON)


@dataclass
class KernelCacheEntry:
    """
    Dataclass/Struct that stores necessary information S.T we can
    reconstruct and call the "cached" kernel.
    """

    namespace: str
    cache_id: str
    kernel_sig: tuple[KernelBufferUsage]
    vmfb: bytes

    @property
    def module_op(self):
        filepath = (
            CACHE_BASE_DIR / self.namespace / self.cache_id / self.cache_id
        ).with_suffix(".mlir")
        with open(filepath, "r") as f:
            module_str = f.read()
        return module_str


NAMESPACE_REGISTRY = Dict[str, "KernelNamespace"] = {}


class KernelNamespace:
    """
    Class for customizing the `get_hash` method for `KernelCacheManager`.
    Instantiating this class will register the namespace with the global `NAMESPACE_REGISTRY`.
    """

    def __init__(self, name: str, hashing_function: Callable):
        self.name = name
        self.get_hash = hashing_function
        self.register()

    def register(self):
        if self.name in NAMESPACE_REGISTRY.keys():
            raise ValueError(
                f"KernelNamespace {self.name} is already registered. Overwriting previous implementation."
            )
        NAMESPACE_REGISTRY[self.name] = self

    @staticmethod
    def add(name, hashing_function: Callable):
        """
        Decorator-based instantiation. Adds the instantiated namespace to global `NAMESPACE_REGISTRY`.
        Example use:

        ```python
        from iree.turbine.kernel.cache import KernelNamespace, NAMESPACE_REGISTRY

        @KernelNamespace.add("my_namespace_name")
        def namespace_specific_hashing_function(...)
            ...

        my_namespace_instance = NAMESPACE_REGISTRY["my_namespace_name"]
        ```
        """

        namespace = KernelNamespace(name, hashing_function)
        return namespace.get_hash


FileCache = Dict[str, set[str]]  # namespace -> set of hashes present in file cache


class KernelCacheManager:
    """
    Generic Kernel cache manager has two main components/cache:

    1. Session/Online cache - This is the main cache that our compiler and runtime will load from and store to. It is
    essentially a dict that uses the kernel hash as keys and the KernelCacheEntry as values. We added LRU functionality with limits
    for number of kernel cached here, because this lives on RAM, and we wouldn't want to run OOM.

    2. File/Offline cache - This cache is essential for loading saved/compiled cache between sessions/runs. This is done
    by storing vital kernel information(vmfb, kernel_sig, and mlir) to CACHE_BASE_DIR/kernel_hash directory. If said kernel
    is queried during a new run and does not exist on session/online cache yet, we'd load files from the kernel_hash directory
    and reconstruct the KernelCacheEntry from it.
    """

    def __init__(self):
        self.namespaces: Dict[str, KernelNamespace] = {}
        self.file_cache: FileCache = {}
        self.session_cache: OrderedDict[str, KernelCacheEntry] = OrderedDict()
        self.lock = threading.Lock()
        self.update_file_cache()

    def get_hash(
        self,
        namespace,
        *args,
        **kwargs,
    ):
        """
        Get a unique identifier for a given kernel. The hashing algorithm is namespace-dependent.
        """
        return self.namespaces[namespace].get_hash(*args, **kwargs)

    ###############################################################################
    # File Cache related helpers
    ###############################################################################

    def update_file_cache(self):
        """
        Search for saved/cached kernels in cache_base_directory and inform
        the cache manager for what are available.
        """
        # Early exit if no cache directory found.
        if not CACHE_BASE_DIR.is_dir():
            return
        for ns_dir in CACHE_BASE_DIR.glob("*/"):
            namespace = ns_dir.name
            if namespace not in self.file_cache.keys():
                self.file_cache[namespace] = {}
                for ker_dir in ns_dir.glob("*/"):
                    hash = ker_dir.name
                    self.file_cache[namespace].add(hash)

    def store_kernel_to_file(
        self,
        namespace,
        kernel_hash,
        vmfb: bytes,
        kernel_sig: tuple[KernelBufferUsage],
        module_str: str,
    ):
        """
        Stores/save compiled kernels into CACHE_BASE_DIR/namespace/kernel_hash
        including it's MLIR, VMFB, and kernel signature.
        """
        cur_cache_dir = CACHE_BASE_DIR / namespace / kernel_hash
        os.makedirs(cur_cache_dir, exist_ok=True)
        cur_cache_basefile = cur_cache_dir / kernel_hash
        cur_vmfb_path = cur_cache_basefile.with_suffix(".vmfb")
        cur_module_path = cur_cache_basefile.with_suffix(".mlir")
        cur_kernelsig_path = cur_cache_basefile.with_suffix(".json")
        _write_file(cur_vmfb_path, "wb", vmfb)
        _write_file(cur_module_path, "w", module_str)
        kernel_sig_str = json.dumps([usage.name for usage in kernel_sig])
        _write_file(cur_kernelsig_path, "w", kernel_sig_str)

    def load_kernel_from_file(self, namespace, kernel_hash):
        """
        Loads the queried kernel(including VMFB, and kernel signature)
        from local cache file/directory.
        """
        cur_cache_dir = CACHE_BASE_DIR / namespace / kernel_hash
        vmfb = None
        kernel_sig_str = None
        if not os.path.exists(cur_cache_dir):
            raise ValueError("Failed to find queried cached kernel.")
        cur_cache_basefile = cur_cache_dir / kernel_hash
        cur_vmfb_path = cur_cache_basefile.with_suffix(".vmfb")
        cur_kernelsig_path = cur_cache_basefile.with_suffix(".json")
        vmfb = _read_file(cur_vmfb_path, "rb")
        kernel_sig_str = json.loads(_read_file(cur_kernelsig_path, "r"))
        kernel_sig = [KernelBufferUsage[usage] for usage in kernel_sig_str]
        return KernelCacheEntry(namespace, kernel_hash, kernel_sig, vmfb)

    ###############################################################################
    # Session cache related helpers
    ###############################################################################
    def store_kernel_to_session(
        self, kernel_hash: str, cached_kernel: KernelCacheEntry
    ):
        """
        LRU style storing of kernel into session cache. Set most recently generated kernel to top of session cache,
        and if len of cache exceed limit, we'd pop least recently used
        """
        self.session_cache[kernel_hash] = cached_kernel
        self.session_cache.move_to_end(kernel_hash)
        if len(self.session_cache) > CACHE_LIMIT:
            self.session_cache.popitem(last=False)

    def store_kernel(
        self,
        vmfb: bytes,
        kernel_sig: tuple[KernelBufferUsage],
        module_str: str,
        namespace: str,
        kernel_hash: str,
    ):
        """
        Save given kernel(vmfb, kernel_sig, and MLIR) into session_cache and file/offline cache.
        """
        if not CACHE_ON or not kernel_hash:
            return
        with self.lock:
            self.store_kernel_to_file(kernel_hash, vmfb, kernel_sig, module_str)
            if not ALWAYS_COMPILE:
                # Do not store in session cache if always compile to save memory.
                self.store_kernel_to_session(
                    kernel_hash,
                    KernelCacheEntry(namespace, kernel_hash, kernel_sig, vmfb),
                )

    def load_kernel(self, namespace, kernel_hash: str):
        """
        LRU style loading of kernel from session cache and move queried kernel to top of LRU if it exist.
        If it only exist in file/offline cache, we'll load from local files, reconstruct KernelCacheEntry and then store
        into session_cache.If it does not exist in session cache nor offline/file cache, then we return "None"
        and ask compiler to compile from scratch.
        """
        if ALWAYS_COMPILE or not kernel_hash or not CACHE_ON:
            return None
        with self.lock:
            if kernel_hash in self.session_cache:
                self.session_cache.move_to_end(kernel_hash)
            elif kernel_hash in self.file_cache[namespace]:
                cached_kernel = self.load_kernel_from_file(kernel_hash)
                self.store_kernel_to_session(kernel_hash, cached_kernel)
            return self.session_cache.get(kernel_hash, None)


def get_cache_manager() -> KernelCacheManager:
    global _global_cache_manager
    if not "_global_cache_manager" in globals():
        _global_cache_manager = KernelCacheManager()
    return _global_cache_manager


def reset_cache_manager() -> KernelCacheManager:
    if not "_global_cache_manager" in globals():
        return
    if os.path.exists(CACHE_BASE_DIR):
        shutil.rmtree(CACHE_BASE_DIR)
    global _global_cache_manager
    del _global_cache_manager


# Cache for the system context and vm function.
RUNTIME_CACHE: dict[str, tuple[rt.SystemContext, rt.VmFunction]] = {}

####
# utility functions for invoking a cached kernel. Needs some work.
####

import iree.runtime as rt
import ctypes


def invoke_cached_kernel(
    cached_kernel: KernelCacheEntry,
    args: list[torch.Tensor],
    config: dict[str, str],
    dynamic_symbols: list[IndexExpr],
    dynamic_symbols_map: dict[IndexExpr, int],
    run: bool,
    run_bench: bool,
    namespace: str,
    kernel_hash: str,
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
        "isolated_benchmark",
        config,
        kernel_inputs,
        kernel_outputs,
        kernel_dynamic_dims,
        run,
        run_bench,
        inplace=True,
        namespace=namespace,
        kernel_hash=kernel_hash,
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

    if inplace:
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


def _read_file(name, mode):
    with open(name, mode) as file:
        data = file.read()
    return data


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
