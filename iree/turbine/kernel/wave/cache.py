# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import copy
import glob
import hashlib
import inspect
import json
import os
import shutil
import threading
import math

from collections import OrderedDict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import functools
from typing import Any, Callable, Optional

from iree.turbine.kernel.lang.kernel_buffer import KernelBufferMeta

from .constraints import Constraint, TilingConstraint, WaveConstraint
from ..compiler.kernel_codegen import KernelBufferUsage
from ..lang.wave_types import IndexMapping
from .utils.classes import KernelLaunchInfo
from .compile_options import WaveCompileOptions

default_cache_base_dir = Path.home() / ".wave"
CACHE_BASE_DIR = Path(os.environ.get("WAVE_CACHE_DIR", default_cache_base_dir))
WAVE_RUNTIME_DIR = CACHE_BASE_DIR / "wave_runtime"
WAVE_ALWAYS_COMPILE = int(os.environ.get("WAVE_ALWAYS_COMPILE", 0))
WAVE_CACHE_ON = int(os.environ.get("WAVE_CACHE_ON", 1))
WAVE_CACHE_LIMIT = int(os.environ.get("WAVE_CACHE_LIMIT", 16))
MAX_LRU_CACHE_SIZE = int(os.environ.get("WAVE_MAX_LRU_CACHE_SIZE", 128))


def is_cache_enabled() -> bool:
    return bool(WAVE_CACHE_ON)


@dataclass
class WaveCache:
    """
    Dataclass/Struct that stores necessary information S.T we can
    reconstruct and call the "cached" kernel.
    """

    kernel_sig: tuple[KernelBufferUsage]
    vmfb: bytes
    asm: str
    kernel_launch_info: Optional[KernelLaunchInfo] = None


def extract_mappings(kernel_fn: Callable):
    """Look for IndexMapping used in the kernel by iterating over freevars."""
    # Handle when kernel_fn.__closure is None.
    closures = kernel_fn.__closure__ or []
    return [
        freevar.cell_contents
        for freevar in closures
        if isinstance(freevar.cell_contents, IndexMapping)
    ]


def extract_arg_types(kernel_fn: Callable):
    """Look for arg types used in the kernel by iterating over arg signature."""
    return [
        (arg.annotation, arg.annotation.physical_layout)
        for arg in inspect.signature(kernel_fn).parameters.values()
        if isinstance(arg.annotation, KernelBufferMeta)
    ]


def extract_free_vars(kernel_fn: Callable):
    """Look for variables that are defined outside the kernel but impact kernel generated. (e.g is_causal, logit_cap, or mfma_variants)"""
    return [
        (k, v)
        for k, v in inspect.getclosurevars(kernel_fn).nonlocals.items()
        if not isinstance(v, (IndexMapping, Callable))
    ]


def get_nested_functions(root_fn: Callable):
    """Simple BFS search to get all sub functions inside a wave kernel."""
    workqueue = deque([root_fn])
    # Using list instead of set because set orders are non
    # deterministic which can mess up hashes.
    fn_list = [root_fn]
    while workqueue:
        cur_fn = workqueue.pop()
        # Add var to workqueue and fn_list freevar that
        # we have not seen before and who's type is a function.
        sub_fns = [
            f
            for f in inspect.getclosurevars(cur_fn).nonlocals.values()
            if inspect.isfunction(f) and f not in fn_list
        ]
        fn_list.extend(sub_fns)
        workqueue.extend(sub_fns)
    return fn_list


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
    by storing vital kernel information(vmfb, kernel_sig, and mlir) to base_dir/kernel_hash directory. If said kernel
    is queried during a new run and does not exist on session/online cache yet, we'd load files from the kernel_hash directory
    and reconstruct the WaveCache from it.
    """

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.file_cache: set[str] = set()
        self.session_cache: OrderedDict[str, WaveCache] = OrderedDict()
        self.lock = threading.Lock()
        self.update_file_cache()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_hash(
        self,
        constraints: list[Constraint],
        kernel_fn: Callable,
        options: WaveCompileOptions,
    ) -> str:
        """
        Get a unique identifier for a given kernel.
        """
        fns = get_nested_functions(kernel_fn)

        # Add subfunctions invariant property to hash.
        arg_dtypes = extract_arg_types(kernel_fn)
        processed_constraints = anonymize_constraints(constraints)
        key = [
            arg_dtypes,
            processed_constraints,
            options.subs,
            options.dynamic_symbols,
            options.schedule,
            options.use_scheduling_barriers,
        ]

        # Add kernel/helper function specific hashes.
        for fn in fns:
            try:
                kernel_src = inspect.getsource(fn)
                free_vars = extract_free_vars(fn)
                index_mappings = extract_mappings(fn)
            except:
                # sets kernel_hash as None if fail to inspect source.
                # We also taught load_kernel and store_kernel to skip
                # if kernel_hash is None.
                return None
            key += [
                kernel_src,
                index_mappings,
                free_vars,
            ]

        # Benchmark related hash
        if options.run_bench:
            key += [options.benchmark_batch_size]
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
        if not self.base_dir.exists():
            return
        for entry in self.base_dir.iterdir():
            if entry.name not in self.file_cache:
                self.file_cache.add(entry.name)

    def store_kernel_to_file(
        self,
        kernel_hash: str,
        vmfb: bytes,
        kernel_sig: tuple[KernelBufferUsage],
        module_str: str,
        kernel_launch_info: KernelLaunchInfo,
    ):
        """
        Stores/save compiled kernels into self.base_dir/kernel_hash
        including it's MLIR, VMFB, and kernel signature. If wave
        runtime is enabled, also copies the hsaco binary and
        stores the kernel launch information.
        """
        cur_cache_dir = self.base_dir / kernel_hash
        os.makedirs(cur_cache_dir, exist_ok=True)
        cur_cache_basefile = cur_cache_dir / kernel_hash
        cur_vmfb_path = cur_cache_basefile.with_suffix(".vmfb")
        cur_module_path = cur_cache_basefile.with_suffix(".mlir")
        cur_kernelsig_path = cur_cache_basefile.with_suffix(".json")
        cur_vmfb_path.write_bytes(vmfb)
        cur_module_path.write_text(module_str)
        kernel_sig_str = json.dumps([usage.name for usage in kernel_sig])
        cur_kernelsig_path.write_text(kernel_sig_str)
        cur_hsaco_path = glob.glob(str(WAVE_RUNTIME_DIR / "*.hsaco"))
        # Copy the hsaco file to the cache directory only if it exists.
        if cur_hsaco_path:
            cur_hsaco_path = cur_hsaco_path[0]
            shutil.copy(cur_hsaco_path, cur_cache_basefile.with_suffix(".hsaco"))
        cur_kernel_info_path = cur_cache_basefile.with_suffix(".kernel_info.json")
        kernel_launch_info_dict = asdict(kernel_launch_info)
        # Lambdas cannot be serialized by json so remove this from the kernel launch info.
        del kernel_launch_info_dict["grid"]
        kernel_info_str = json.dumps(kernel_launch_info_dict)
        cur_kernel_info_path.write_text(kernel_info_str)

    # This is a static method with the base directory passed in explicitly so
    # that the lru_cache doesn't prevent garbage collection of instances.
    @staticmethod
    @functools.lru_cache
    def load_kernel_from_file(base_dir, kernel_hash):
        """
        Loads the queried kernel(including VMFB, and kernel signature)
        from local cache file/directory.
        """
        cur_cache_dir = base_dir / kernel_hash
        vmfb = None
        kernel_sig_str = None
        if not os.path.exists(cur_cache_dir):
            raise ValueError("Failed to find queried cached kernel.")
        cur_cache_basefile = cur_cache_dir / kernel_hash
        cur_vmfb_path = cur_cache_basefile.with_suffix(".vmfb")
        cur_kernelsig_path = cur_cache_basefile.with_suffix(".json")
        cur_asm_path = cur_cache_basefile.with_suffix(".mlir")
        vmfb = cur_vmfb_path.read_bytes()
        kernel_sig_str = json.loads(cur_kernelsig_path.read_text())
        kernel_sig = [KernelBufferUsage[usage] for usage in kernel_sig_str]
        asm = cur_asm_path.read_text()
        cur_kernel_info_path = cur_cache_basefile.with_suffix(".kernel_info.json")
        kernel_info_str = json.loads(cur_kernel_info_path.read_text())
        # Convert string to lambda. This could have a math dependency
        # and so we include it above.
        kernel_info_str["grid"] = eval(kernel_info_str["grid_str"])
        kernel_launch_info = KernelLaunchInfo(**kernel_info_str)
        return WaveCache(kernel_sig, vmfb, asm, kernel_launch_info)

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
        module_str: str,
        options: WaveCompileOptions,
    ):
        """
        Save given kernel(vmfb, kernel_sig, and MLIR) into session_cache and file/offline cache.
        """
        if not WAVE_CACHE_ON or not options.kernel_hash:
            return
        with self.lock:
            self.store_kernel_to_file(
                options.kernel_hash,
                vmfb,
                options.kernel_usages,
                module_str,
                options.kernel_launch_info,
            )
            if not WAVE_ALWAYS_COMPILE:
                # Do not store in session cache if always compile to save memory.
                self.store_kernel_to_session(
                    options.kernel_hash,
                    WaveCache(
                        options.kernel_usages,
                        vmfb,
                        module_str,
                        options.kernel_launch_info,
                    ),
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
                self.cache_hits += 1
            elif kernel_hash in self.file_cache:
                cached_kernel = self.load_kernel_from_file(self.base_dir, kernel_hash)
                self.store_kernel_to_session(kernel_hash, cached_kernel)
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            return self.session_cache.get(kernel_hash, None)


def get_cache_manager() -> WaveCacheManager:
    global _global_cache_manager
    if not "_global_cache_manager" in globals():
        _global_cache_manager = WaveCacheManager(CACHE_BASE_DIR)
    return _global_cache_manager


def reset_cache_manager(base_dir):
    global _global_cache_manager
    _global_cache_manager = WaveCacheManager(base_dir)
