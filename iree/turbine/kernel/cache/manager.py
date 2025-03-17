# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import json
import os
import shutil
import threading

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from ..compiler.kernel_codegen import KernelBufferUsage

__all__ = [
    "is_cache_enabled",
    "KernelCacheEntry",
    "NAMESPACE_REGISTRY",
    "KernelNamespace",
    "KernelCacheManager",
]

# Fetch enviornment variables.
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


def _read_file(name, mode):
    with open(name, mode) as file:
        data = file.read()
    return data


def _write_file(name, mode, data):
    with open(name, mode) as file:
        file.write(data)


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

    @staticmethod
    def get_cache_manager() -> "KernelCacheManager":
        global _global_cache_manager
        if not "_global_cache_manager" in globals():
            _global_cache_manager = KernelCacheManager()
        return _global_cache_manager

    @staticmethod
    def reset_cache_manager() -> "KernelCacheManager":
        if not "_global_cache_manager" in globals():
            return
        if os.path.exists(CACHE_BASE_DIR):
            shutil.rmtree(CACHE_BASE_DIR)
        global _global_cache_manager
        del _global_cache_manager
