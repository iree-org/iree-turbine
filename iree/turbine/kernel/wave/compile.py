from typing import Any, Optional
import subprocess

import torch
import glob
from copy import copy
from .._support.indexing import IndexingContext
from .._support.location_config import LocationCaptureLevel
from ..compiler import kernel_codegen, host_codegen
from .compile_options import WaveCompileOptions

from .cache import (
    get_cache_base_dir,
    get_cache_manager,
    get_temp_binary_dir,
    is_cache_enabled,
)
from .utils.compile_utils import compile_to_vmfb
from .utils.run_utils import invoke_vmfb, _write_file
from iree.turbine.kernel._support.context import push, pop


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

        invoke_vmfb(
            self.executable, self.options, kernel_inputs, kernel_outputs, self.gpu_func
        )
        return self.asm


from ..compiler.ir import (
    stream_d,
    gpu_d,
    Operation,
    InsertionPoint,
    Context,
    Location,
    WalkResult,
    Module,
    FunctionType,
    BlockArgument,
    TypeAttr,
)


def _deerie(module):
    module = Module.parse(module.operation.get_asm(), context=module.context)

    def find_single_nested(name, parent):
        captured = None
        for op in parent.regions[0].blocks[0].operations:
            # Dynamic typing is hard: must to op.operaiton.name in case some specific class has .name that has a different meaning.
            if op.operation.name == name:
                if captured:
                    raise RuntimeError(f"more than one '{name}' operation found")
                captured = op
        if not captured:
            raise RuntimeError(f"no {name} operation found")
        return captured

    executable = find_single_nested("stream.executable", module.operation)
    local_module = find_single_nested("builtin.module", executable)
    func = find_single_nested("func.func", local_module)

    # TODO: add launch bounds

    to_delete = []  # type: list[Operation]
    subspans = []  # type: list[stream_d.BindingSubspanOp]

    def replace_ids_and_collect_subspans(op: Operation):
        if isinstance(op.opview, stream_d.DispatchWorkgroupIDOp):
            dispatch = op.opview  # type: stream_d.DispatchWorkgroupIDOp
            match dispatch.dimension.value:
                case 0:
                    dimension = gpu_d.Dimension.x
                case 1:
                    dimension = gpu_d.Dimension.y
                case 2:
                    dimension = gpu_d.Dimension.z
            with InsertionPoint(op):
                block_id = gpu_d.BlockIdOp(dimension, loc=op.location)
            op.result.replace_all_uses_with(block_id.result)
            to_delete.append(op)
            return WalkResult.ADVANCE

        if isinstance(op.opview, stream_d.BindingSubspanOp):
            subspan = op.opview  # type: stream_d.BindingSubspanOp
            subspans.append(subspan)

        return WalkResult.ADVANCE

    func.walk(replace_ids_and_collect_subspans)
    old_func_type = func.attributes["function_type"].value
    func_input_types = old_func_type.inputs
    for subspan in subspans:
        subspan.binding.set_type(subspan.result.type)
        func_input_types[
            BlockArgument(subspan.binding).arg_number
        ] = subspan.result.type
        subspan.result.replace_all_uses_except(subspan.binding, subspan.operation)
        to_delete.append(subspan)
    func.attributes["function_type"] = TypeAttr.get(
        FunctionType.get(
            func_input_types, old_func_type.results, context=old_func_type.context
        ),
        context=old_func_type.context,
    )

    for op in to_delete:
        op.erase()

    return local_module.get_asm(binary=False, print_generic_op_form=True)


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
                options, cached_kernel.vmfb, cached_kernel.asm, binary_path
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
    )
    asm = mb.module_op.get_asm(
        enable_debug_info=options.location_capture_config.level
        != LocationCaptureLevel.NONE,
        use_local_scope=options.use_local_scope,
    )
    if options.print_mlir:
        print(asm)

    if options.use_water_leak_check:
        print()
        try:
            from water_mlir import binaries as water_bin
        except ImportError as err:
            raise RuntimeError("optional water_mlir module not installed") from err
        binary = water_bin.find_binary("water-opt")
        generic_mlir = _deerie(mb.module_op)
        result = subprocess.run(
            [
                binary,
                "-allow-unregistered-dialect",
                "--pass-pipeline=builtin.module(water-assert-in-bounds{include-vector-load-store=1 create-speculative-funcs=1})",
            ],
            input=generic_mlir,
            capture_output=True,
            text=True,
        )
        print(str(result.stdout))
        print(str(result.stderr))

    if options.override_mlir:
        asm = options.override_mlir

    if options.compile_to_mlir:
        return WaveKernel(options, None, asm, None)

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

    return WaveKernel(options, compiled_wave_vmfb, asm, binary_path)
