from typing import Any

import torch
from copy import copy
from .._support.indexing import IndexingContext, IndexExpr
from ..compiler import kernel_codegen, host_codegen
from .compile_options import WaveCompileOptions

from .cache import (
    is_cache_enabled,
    get_cache_manager,
    WAVE_RUNTIME_DIR,
)
from .utils.compile_utils import compile_to_vmfb
from .utils.run_utils import invoke_vmfb, _write_file
from iree.turbine.kernel._support.context import push, pop


class WaveKernel:
    """
    Represents a wave kernel that can be invoked by the user.
    """

    def __init__(self, options: WaveCompileOptions, executable: Any, asm: str):
        self.options = options
        self.executable = executable
        self.asm = asm

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

        kernel_inputs.extend(scalar_args)

        invoke_vmfb(self.executable, self.options, kernel_inputs, kernel_outputs)
        return self.asm


def wave_compile(options: WaveCompileOptions, kernel: "LaunchableWave") -> WaveKernel:
    """
    Compiles the wave kernel to an executable.
    """

    # Check if this kernel has been compiled before, if the cache is enabled.
    cache_manager = None
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
            return WaveKernel(options, cached_kernel.vmfb, cached_kernel.asm)

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
        options.dump_binaries = str(WAVE_RUNTIME_DIR)

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
        mb, exe, kernel_sig, entrypoint_name, options.dynamic_symbols
    )
    asm = mb.module_op.get_asm(
        enable_debug_info=options.debug_info, use_local_scope=options.use_local_scope
    )
    if options.print_mlir:
        print(asm)

    if options.override_mlir:
        asm = options.override_mlir

    if options.compile_to_mlir:
        return WaveKernel(options, None, asm)

    print(asm)
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

    return WaveKernel(options, compiled_wave_vmfb, asm)
