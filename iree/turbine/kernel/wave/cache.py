# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import copy
import hashlib
import inspect
import json
import os
import shutil
import torch
import threading

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constraints import Constraint, TilingConstraint, WaveConstraint
from ..compiler.kernel_codegen import KernelBufferUsage
from ..lang.wave_types import IndexMapping
from .._support.indexing import IndexExpr
from .utils import invoke_vmfb, _read_file, _write_file

default_cache_base_dir = Path.home() / ".wave"
CACHE_BASE_DIR = Path(os.environ.get("WAVE_CACHE_DIR", default_cache_base_dir))
WAVE_ALWAYS_COMPILE = int(os.environ.get("WAVE_ALWAYS_COMPILE", 0))
WAVE_CACHE_ON = int(os.environ.get("WAVE_CACHE_ON", 1))
WAVE_CACHE_LIMIT = int(os.environ.get("WAVE_CACHE_LIMIT", 16))


def is_cache_enabled() -> bool:
    return bool(WAVE_CACHE_ON)


@dataclass
class WaveCache:
    """
    Dataclass/Struct that stores necessary information S.T we can
    reconstruct and call the "cached" kernel.
    """

    cache_id: str
    kernel_sig: tuple[KernelBufferUsage]
    vmfb: bytes

    @property
    def module_op(self):
        filepath = (CACHE_BASE_DIR / self.cache_id / self.cache_id).with_suffix(".mlir")
        with open(filepath, "r") as f:
            module_str = f.read()
        return module_str


def extract_mappings(kernel_fn: Callable):
    """Look for IndexMapping used in the kernel by iterating over freevars."""
    return [
        freevar.cell_contents
        for freevar in kernel_fn.__closure__
        if isinstance(freevar.cell_contents, IndexMapping)
    ]


def extract_arg_types(kernel_fn: Callable):
    """Look for arg types used in the kernel by iterating over arg signature."""
    return [arg.annotation for arg in inspect.signature(kernel_fn).parameters.values()]


def anonymize_constraints(input_constraints: list[Constraint]):
    """
    Helper function to anonymize constraint S.T we can have the same generate
    hash before and after initializing constraints and induction variables.

    This is crucial to enable kernels being called under same LaunchableWave have
    the same kernel cache despite having constraints and iv initialized.

    Note that this annonymization would not affect the correctness of the hash,
    because the factors that can impact initialization of these constraints exist
    in different parts of the hash.
    """
    processed_constraints = copy.deepcopy(input_constraints)
    for constraint in processed_constraints:
        if isinstance(constraint, TilingConstraint):
            constraint.induction_var = None
        elif isinstance(constraint, WaveConstraint):
            constraint.wave_id = None
        else:
            continue


class WaveCacheManager(object):
    """
    Wave cache manager has two main components/cache:

    1. Session/Online cache - This is the main cache that our compiler and runtime will load from and store to. It is
    essentially a dict that uses the kernel hash as keys and the WaveCache as values. We added LRU functionality with limits
    for number of kernel cached here, because this lives on RAM, and we wouldn't want to run OOM.

    2. File/Offline cache - This cache is essential for loading saved/compiled cache between sessions/runs. This is done
    by storing vital kernel information(vmfb, kernel_sig, and mlir) to CACHE_BASE_DIR/kernel_hash directory. If said kernel
    is queried during a new run and does not exist on session/online cache yet, we'd load files from the kernel_hash directory
    and reconstruct the WaveCache from it.
    """

    def __init__(self):
        self.file_cache: set[str] = set()
        self.session_cache: OrderedDict[str, WaveCache] = OrderedDict()
        self.lock = threading.Lock()
        self.update_file_cache()

    def get_hash(
        self,
        constraints: list[Constraint],
        kernel_fn: Callable,
        hyperparams: dict[IndexExpr, Any],
        dynamic_symbols: list[IndexExpr, Any],
        config: dict[str, str],
        use_scheduling: bool,
        use_scheduling_barriers: bool,
        run_bench: bool,
    ):
        """
        Get a unique identifier for a given kernel.
        """
        try:
            kernel_src = inspect.getsource(kernel_fn)
            index_mappings = extract_mappings(kernel_fn)
            arg_dtypes = extract_arg_types(kernel_fn)
        except:
            # sets kernel_hash as None if fail to inspect source.
            # We also taught load_kernel and store_kernel to skip
            # if kernel_hash is None.
            return None
        processed_constraints = anonymize_constraints(constraints)
        key = [
            kernel_src,
            processed_constraints,
            hyperparams,
            dynamic_symbols,
            use_scheduling,
            use_scheduling_barriers,
            index_mappings,
            arg_dtypes,
        ]

        # Benchmark related hash
        if run_bench and config != None:
            key += config.get("benchmark_batch_size", "")
        return hashlib.sha256(str(key).encode("utf-8")).hexdigest()

    ###############################################################################
    # File Cache related helpers
    ###############################################################################

    def update_file_cache(self):
        """
        Search for saved/cached kernels in cache_base_directory and inform
        the cache manager for what are available.
        """
        # Early exit if no cache directory found.
        if not os.path.exists(CACHE_BASE_DIR):
            return
        for entry in os.scandir(CACHE_BASE_DIR):
            if entry.name not in self.file_cache:
                self.file_cache.add(entry.name)

    def store_kernel_to_file(
        self,
        kernel_hash,
        vmfb: bytes,
        kernel_sig: tuple[KernelBufferUsage],
        module_str: str,
    ):
        """
        Stores/save compiled kernels into CACHE_BASE_DIR/kernel_hash
        including it's MLIR, VMFB, and kernel signature.
        """
        cur_cache_dir = CACHE_BASE_DIR / kernel_hash
        if not os.path.exists(cur_cache_dir):
            os.makedirs(cur_cache_dir)
        cur_cache_basefile = cur_cache_dir / kernel_hash
        cur_vmfb_path = cur_cache_basefile.with_suffix(".vmfb")
        cur_module_path = cur_cache_basefile.with_suffix(".mlir")
        cur_kernelsig_path = cur_cache_basefile.with_suffix(".json")
        _write_file(cur_vmfb_path, "wb", vmfb)
        _write_file(cur_module_path, "w", module_str)
        kernel_sig_str = json.dumps([usage.name for usage in kernel_sig])
        _write_file(cur_kernelsig_path, "w", kernel_sig_str)

    def load_kernel_from_file(self, kernel_hash):
        """
        Loads the queried kernel(including VMFB, and kernel signature)
        from local cache file/directory.
        """
        cur_cache_dir = CACHE_BASE_DIR / kernel_hash
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
        return WaveCache(kernel_hash, kernel_sig, vmfb)

    ###############################################################################
    # Session cache related helpers
    ###############################################################################
    def store_kernel_to_session(self, kernel_hash: str, cached_kernel: WaveCache):
        """
        LRU style storing of kernel into session cache. Set most recently generated kernel to top of session cache,
        and if len of cache exceed limit, we'd pop least recently used
        """
        self.session_cache[kernel_hash] = cached_kernel
        self.session_cache.move_to_end(kernel_hash)
        if len(self.session_cache) > WAVE_CACHE_LIMIT:
            self.session_cache.popitem(last=False)

    def store_kernel(
        self,
        vmfb: bytes,
        kernel_sig: tuple[KernelBufferUsage],
        module_str: str,
        kernel_hash: str,
    ):
        """
        Save given kernel(vmfb, kernel_sig, and MLIR) into session_cache and file/offline cache.
        """
        if not WAVE_CACHE_ON or not kernel_hash:
            return
        with self.lock:
            self.store_kernel_to_file(kernel_hash, vmfb, kernel_sig, module_str)
            if not WAVE_ALWAYS_COMPILE:
                # Do not store in session cache if always compile to save memory.
                self.store_kernel_to_session(
                    kernel_hash, WaveCache(kernel_hash, kernel_sig, vmfb)
                )

    def load_kernel(self, kernel_hash: str):
        """
        LRU style loading of kernel from session cache and move queried kernel to top of LRU if it exist.
        If it only exist in file/offline cache, we'll load from local files, reconstruct WaveCache and then store
        into session_cache.If it does not exist in session cache nor offline/file cache, then we return "None"
        and ask compiler to compile from scratch.
        """
        if WAVE_ALWAYS_COMPILE or not kernel_hash or not WAVE_CACHE_ON:
            return None
        with self.lock:
            if kernel_hash in self.session_cache:
                self.session_cache.move_to_end(kernel_hash)
            elif kernel_hash in self.file_cache:
                cached_kernel = self.load_kernel_from_file(kernel_hash)
                self.store_kernel_to_session(kernel_hash, cached_kernel)
            return self.session_cache.get(kernel_hash, None)


def get_cache_manager() -> WaveCacheManager:
    global _global_cache_manager
    if not "_global_cache_manager" in globals():
        _global_cache_manager = WaveCacheManager()
    return _global_cache_manager


def reset_cache_manager() -> WaveCacheManager:
    if not "_global_cache_manager" in globals():
        return
    if os.path.exists(CACHE_BASE_DIR):
        shutil.rmtree(CACHE_BASE_DIR)
    global _global_cache_manager
    del _global_cache_manager


def invoke_cached_kernel(
    cached_kernel: WaveCache,
    args: list[torch.Tensor],
    config: dict[str, str],
    dynamic_symbols: list[IndexExpr],
    dynamic_symbols_map: dict[IndexExpr, int],
    run: bool,
    run_bench: bool,
):
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

    if not config:
        raise ValueError("no config provided")

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
    )
