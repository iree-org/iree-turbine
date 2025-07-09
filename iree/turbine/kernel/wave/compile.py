from typing import Any, Optional

import torch
import glob
from copy import copy
from .._support.indexing import IndexingContext
from .._support.location_config import LocationCaptureLevel
from ..compiler import kernel_codegen, host_codegen
from .compile_options import WaveCompileOptions
from .water import water_leak_in_bounds_check

from .cache import (
    get_cache_base_dir,
    get_cache_manager,
    get_temp_binary_dir,
    is_cache_enabled,
)
from .utils.compile_utils import compile_to_vmfb
from .utils.run_utils import print_bench_result, write_file, invoke_with_wave_runtime
from .profiling import benchmark_module
from iree.turbine.kernel._support.context import push, pop
from iree.turbine.kernel.lang import IndexSymbol
from iree.turbine.runtime.launch import Launchable
import iree.runtime as rt


class WaveKernel:
    """
    Represents a wave kernel that can be invoked by the user.
    """

    def __init__(
        self,
        options: WaveCompileOptions,
        executable: Any,
        asm: str,
        gpu_binary_path: Optional[str],
        bound_scalar_symbols: dict[IndexSymbol, int],
        symbols_args_map: dict[IndexSymbol, tuple[int, int]],
    ):
        self.options = options
        self.executable = executable
        self.asm = asm
        if gpu_binary_path:
            import wave_runtime

            self.gpu_binary, self.gpu_func = wave_runtime.load_binary(
                gpu_binary_path, options.kernel_launch_info.func_name
            )
        else:
            self.gpu_func = None
        self.bound_scalar_symbols = bound_scalar_symbols
        self.symbols_args_map = symbols_args_map

        if not options.wave_runtime:
            # 'launchable' decides if function is async or not based on name.
            self.func_name = options.func_name + (
                "$async" if options.iree_launch_async else ""
            )

            def loader(device):
                vm_instance = device.vm_instance
                return rt.VmModule.copy_buffer(vm_instance, self.executable)

            self.launchable = Launchable.from_vm_module(
                loader,
                entry_point=self.func_name,
            )

        if options.profile_python_wrapper:
            self.call_handler = self.invoke_with_profile
        else:
            self.call_handler = self.invoke

    def __call__(self, *args, **kwargs):
        return self.call_handler(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        """
        Invokes the wave kernel with the given arguments.
        Returns the assembly code of the compiled kernel.
        """

        # Segregate args into kernel tensor and scalars.
        scalar_args = []
        kernel_inputs, kernel_outputs = [], []

        # Partition arguments into kernel inputs and outputs.
        # ToDo: we should expose the `usage` as a property in binding desc
        #       so that we can reduce the code and use `zip``.
        usage_idx = 0
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                scalar_args.append(arg)
                continue
            usage = self.options.kernel_usages[usage_idx]
            usage_idx += 1
            if usage == kernel_codegen.KernelBufferUsage.INPUT:
                kernel_inputs.append(arg)
            if usage == kernel_codegen.KernelBufferUsage.OUTPUT:
                kernel_outputs.append(arg)

        dynamic_symbols = []
        for sym in self.options.dynamic_symbols:
            arg_idx, dim = self.symbols_args_map[sym]
            dynamic_symbols.append(args[arg_idx].shape[dim])

        if self.options.wave_runtime:
            invoke_with_wave_runtime(
                self.options,
                kernel_inputs,
                kernel_outputs,
                scalar_args,
                self.bound_scalar_symbols,
                dynamic_symbols,
                self.gpu_func,
            )
        else:
            self.launchable(
                *kernel_inputs, *kernel_outputs, *scalar_args, *dynamic_symbols
            )

            if self.options.run_bench:
                benchmark_flags = {}
                benchmark_flags["batch_size"] = self.options.benchmark_batch_size

                if self.options.benchmark_repetitions is not None:
                    benchmark_flags["benchmark_repetitions"] = int(
                        self.options.benchmark_repetitions
                    )
                benchmark_results = benchmark_module(
                    self.options,
                    kernel_inputs,
                    kernel_outputs,
                    dynamic_symbols,
                    self.executable,
                    self.func_name,
                    **benchmark_flags,
                )
                print_bench_result(benchmark_results, self.options.bench_file)

        return self.asm

    def invoke_with_profile(self, *args, **kwargs):

        # Warmup
        for _ in range(self.options.profile_python_warmup):
            self.invoke(*args, **kwargs)

        repetitions = self.options.profile_python_repetitions
        if self.options.profile_python_cprofile:
            import cProfile

            with cProfile.Profile() as pr:
                for _ in range(repetitions):
                    res = self.invoke(*args, **kwargs)

            pr.print_stats(sort="cumulative")
            return res
        else:
            import timeit

            time = timeit.timeit(
                lambda: self.invoke(*args, **kwargs),
                number=repetitions,
            )
            print(f"Time: {time:.3f}s, {time / repetitions:.6f}s per iteration")
            return self.invoke(*args, **kwargs)


def wave_compile(options: WaveCompileOptions, kernel: "LaunchableWave") -> WaveKernel:
    """
    Compiles the wave kernel to an executable.
    """

    # Check if this kernel has been compiled before, if the cache is enabled.
    cache_manager = None
    binary_path = None

    def get_binary_path():
        if is_cache_enabled():
            return (
                str(get_cache_base_dir() / options.kernel_hash / options.kernel_hash)
                + ".hsaco"
            )
        else:
            return glob.glob(str(get_temp_binary_dir() / "*.hsaco"))[0]

    bound_scalar_symbols = kernel.bound_scalar_symbols
    symbols_args_map = kernel.symbols_args_map
    if is_cache_enabled():
        cache_manager = get_cache_manager()
        options.kernel_hash = cache_manager.get_hash(
            kernel.constraints,
            kernel._f,
            options,
        )
        cached_kernel = cache_manager.load_kernel(options.kernel_hash)
        if cached_kernel:
            options.kernel_usages = cached_kernel.kernel_sig
            options.kernel_launch_info = cached_kernel.kernel_launch_info
            if options.wave_runtime:
                binary_path = get_binary_path()
            return WaveKernel(
                options,
                cached_kernel.vmfb,
                cached_kernel.asm,
                binary_path,
                bound_scalar_symbols,
                symbols_args_map,
            )

    # Create an indexing context and populate substitutions.
    push(IndexingContext, IndexingContext())
    idxc = IndexingContext.current()

    # Make a copy of the substitutions to avoid mutating the original
    # options.subs.
    idxc.subs = copy(options.subs)

    # For the wave runtime, we need the hsaco binary. So we turn on
    # dumping of binaries and store in wave runtime directory. If we
    # are caching, this will be moved to the appropriate directory.
    if options.wave_runtime:
        options.dump_binaries = get_temp_binary_dir()

    # Recompile kernel from scratch if not found in cache.
    (
        mb,
        graph,
        exe,
        kernel_sig,
        entrypoint_name,
        options,
    ) = kernel._trace_and_get_kernel_signature(options)
    options.kernel_sig = kernel_sig

    host_codegen.isolated_test_call(
        mb,
        exe,
        kernel_sig,
        entrypoint_name,
        options.func_name,
        options.dynamic_symbols,
        location_capture_config=options.location_capture_config,
        async_dispatch=options.iree_launch_async,
    )
    asm = mb.module_op.get_asm(
        enable_debug_info=options.location_capture_config.level
        != LocationCaptureLevel.NONE,
        use_local_scope=options.use_local_scope,
    )
    if options.print_mlir:
        print(asm)

    if options.use_water_leak_check:
        water_leak_in_bounds_check(mb.module_op)

    if options.override_mlir:
        asm = options.override_mlir

    if options.compile_to_mlir:
        return WaveKernel(
            options, None, asm, None, bound_scalar_symbols, symbols_args_map
        )

    compiled_wave_vmfb = compile_to_vmfb(asm, options)
    if options.create_vmfb_file:
        write_file(options.create_vmfb_file, "wb", compiled_wave_vmfb)

    kernel_usages = [
        binding.kernel_buffer_type.usage
        for binding in kernel_sig.kernel_buffer_bindings
    ]
    options.kernel_usages = kernel_usages

    if is_cache_enabled():
        cache_manager.store_kernel(
            compiled_wave_vmfb,
            asm,
            options,
        )

    # Remove the indexing context.
    pop(IndexingContext)
    if options.wave_runtime:
        binary_path = get_binary_path()

    return WaveKernel(
        options,
        compiled_wave_vmfb,
        asm,
        binary_path,
        bound_scalar_symbols,
        symbols_args_map,
    )
