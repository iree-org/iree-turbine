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
from .utils.run_utils import invoke_vmfb, _write_file
from iree.turbine.kernel._support.context import push, pop
from iree.turbine.kernel.lang import IndexSymbol


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
        trace: Optional["CapturedTrace"] = None,
    ):
        self.options = options
        self.executable = executable
        self.asm = asm
        self.trace = trace
        if gpu_binary_path:
            import wave_runtime

            # It is safe to call this multiple times.
            wave_runtime.load_hip_functions()

            self.gpu_binary, self.gpu_func = wave_runtime.load_binary(
                gpu_binary_path, options.kernel_launch_info.func_name
            )
        else:
            self.gpu_func = None
        self.bound_scalar_symbols = bound_scalar_symbols
        self.symbols_args_map = symbols_args_map

    def get_trace(self) -> Optional["CapturedTrace"]:
        """Returns the trace used to generate this kernel.
        If this is a cached kernel, the trace is not available.
        """
        return self.trace

    def __call__(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)

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

        invoke_vmfb(
            self.executable,
            self.options,
            kernel_inputs,
            kernel_outputs,
            scalar_args,
            self.bound_scalar_symbols,
            dynamic_symbols,
            self.gpu_func,
        )
        return self.asm


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

            if options.print_mlir:
                print(cached_kernel.asm)

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

    # Get the trace from the kernel. Since the trace contains complex objects
    # that are not easily serializable, we don't cache the trace. So this trace
    # is not available for cached kernels. The primary use case for the trace is
    # is for tuning where each kernel is different from the others and so we
    # don't want to cache the kernel in that case.
    trace = kernel._trace()

    host_codegen.isolated_test_call(
        mb,
        exe,
        kernel_sig,
        entrypoint_name,
        options.func_name,
        options.dynamic_symbols,
        location_capture_config=options.location_capture_config,
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
        _write_file(options.create_vmfb_file, "wb", compiled_wave_vmfb)

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
        trace,
    )
