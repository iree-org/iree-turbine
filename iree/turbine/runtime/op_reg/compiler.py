# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from timeit import default_timer
from typing import Any, Optional

from iree.compiler.api import (
    Session,
    Source,
    Output,
)

from iree.runtime import (
    VmContext,
    VmFunction,
    VmModule,
)

from ...support.exceptions import (
    GeneralError,
)

from ...support.ir_imports import (
    Attribute,
    Location,
    PassManager,
    SymbolTable,
)

from ...support.logging import (
    runtime_logger as logger,
)

from ..device import (
    Device,
)

from ..tracing import tracer

from .base import (
    FreeFuncKernelBuilder,
    KernelSelection,
)


@dataclass(slots=True)
class KernelCompileConfig:
    # Unique key for this kernel.
    key: str

    # Compiler flags to pass.
    flags: list[str]

    # Use the in-process compiler (default). Some compiler options are only
    # available when invoked standalone/out-of-process, so this is allowed.
    # Out-of-process can also be a useful debugging feature and may be
    # globally controlled.
    in_process: bool = True

    # Whether compiled for async invocations.
    async_invocations: bool = False

    # Whether we compiled with layout specialization and can handle certain
    # permutations of strided tensors. This is currently not supported but will
    # be at some point. Having the option lets us annotate code paths that are
    # NYI.
    layout_specialized: bool = False

    # Arbitrary objects to keep alive as part of this config. This can include
    # things like unbacked memory mappings, etc.
    keep_alive: Any = None

    # If tracing is enabled, this may contain a sanitized key that can be
    # used to log additional information against the kernel.
    tracing_key: Optional[str] = None


# TODO: The cache should be more than just a simple dict. Can be persistent
KERNEL_CACHE: dict[str, tuple[VmContext, VmFunction, KernelCompileConfig]] = {}
MODULE_CACHE: dict[str, tuple[VmModule, KernelCompileConfig]] = {}


def _testing_get_cache_size() -> int:
    return len(KERNEL_CACHE)


def _testing_get_sub_cache_size() -> int:
    return len(MODULE_CACHE)


def assemble_invocable(
    call_func: str, vm_module: VmModule, device: Device
) -> tuple[VmContext, VmFunction]:
    """Creates a VmContext and returns it with a VmFuncton extracted by name from the given vm module."""
    vm_instance = device.vm_instance
    vm_context = VmContext(vm_instance, [device.create_hal_module(), vm_module])
    main_function = vm_module.lookup_function(call_func)
    return vm_context, main_function


def compile_standalone_kernel(
    device: Device,
    ksel: KernelSelection,
    func_name: str = "main",
    async_invocations: bool = True,
) -> tuple[VmContext, VmFunction, KernelCompileConfig]:
    # Early exit on full cache hit.
    cache_key = f"{ksel.spec_key}::{device.instance_cache_key}"
    cache_hit = KERNEL_CACHE.get(cache_key)
    if cache_hit is not None:
        return cache_hit

    # Try to get at least a VMFB cache hit.
    # This sub cache avoids re-compilation for separate gpu's or cuda streams.
    sub_cache_key = f"{ksel.spec_key}::{device.type_cache_key}"
    sub_cache_hit = MODULE_CACHE.get(sub_cache_key)
    if sub_cache_hit is not None:
        vm_module, config = sub_cache_hit
        call_func = f"{func_name}$async" if config.async_invocations else func_name
        vm_context, vm_function = assemble_invocable(call_func, vm_module, device)
        cache_hit = (vm_context, vm_function, config)
        KERNEL_CACHE[cache_key] = cache_hit
        return vm_context, vm_function, config

    # Cache miss.
    start = default_timer()
    config = KernelCompileConfig(
        cache_key,
        list(device.compile_target_flags),
        async_invocations=async_invocations,
    )
    kb = FreeFuncKernelBuilder.create_module(ksel, func_name=func_name)
    with kb.ip, Location.unknown():
        ksel.op.generate(ksel, kb)

    symb = func_name
    if config.async_invocations:
        pm = PassManager.parse("builtin.module(torch-iree-func-conversion)", kb.context)
        pm.run(kb.module_op)
        symb += "$async"

    func = SymbolTable(kb.module_op)[symb]
    if kb.ksel.op.single_dispatch:
        pipeline_attr = Attribute.parse(
            '#util.preprocessing_pipeline<"iree-preprocessing-make-single-dispatch">',
            kb.context,
        )
        func.attributes["preprocessing_pipeline"] = pipeline_attr

    kb.module_op.verify()
    # DO NOT SUBMIT: https://github.com/iree-org/iree/issues/17132
    enable_debug_info = False
    module_asm = kb.module_op.get_asm(
        binary=True, enable_debug_info=enable_debug_info, print_generic_op_form=False
    )
    generation_time = default_timer() - start

    if not config.in_process:
        raise NotImplementedError("Out-of-process compilation not yet supported")

    # TODO: We could be caching the session per device type key.
    # TODO: Create the source and get the module to build into from that vs
    # reserializing (once issues are worked out for that).
    start = default_timer()
    session = Session()
    session.set_flags(*config.flags)
    inv = session.invocation()
    source = Source.wrap_buffer(session, module_asm)
    output = Output.open_membuffer()
    inv.enable_console_diagnostics()
    inv.parse_source(source)
    if not inv.execute():
        # TODO: Capture diagnostics and report.
        raise GeneralError(f"Kernel compilation failed. See diagnostics.")
    inv.output_vm_bytecode(output)
    mapped_memory = output.map_memory()
    compilation_time = default_timer() - start

    # Load.
    vm_instance = device.vm_instance
    # TODO: VmModule.wrap_buffer would be better here, but it is still
    # unreliable capturing mapped memory from the compiler.
    # See: https://github.com/iree-org/iree/issues/17403
    vm_module = VmModule.copy_buffer(vm_instance, mapped_memory)

    sub_cache_hit = (vm_module, config)
    MODULE_CACHE[sub_cache_key] = sub_cache_hit

    call_func = f"{func_name}$async" if config.async_invocations else func_name
    # TODO: We should be able to wrap the buffer as below but there are some
    # subtle ref-counting/shutdown sequencing issues that need to be resolved.
    # vm_module = VmModule.wrap_buffer(vm_instance, mapped_memory)
    vm_context, main_function = assemble_invocable(call_func, vm_module, device)

    if tracer.enabled:
        config.tracing_key = tracer.save_jit_kernel_artifacts(
            cache_key=cache_key, module_asm=module_asm, binary=mapped_memory
        )
        tracer.log_structured(
            tag="COMPILE",
            msg=f"Compiled kernel {config.tracing_key}, cache_key={cache_key}",
            columns=[
                config.tracing_key,
                main_function.name,
                len(module_asm),
                len(mapped_memory),
                generation_time * 1000,
                compilation_time * 1000,
                " ".join(session.get_flags(non_default_only=True)),
            ],
        )
    cache_hit = (vm_context, main_function, config)
    KERNEL_CACHE[cache_key] = cache_hit
    return cache_hit
