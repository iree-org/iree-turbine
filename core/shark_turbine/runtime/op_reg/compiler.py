# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from timeit import default_timer
from typing import Any

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
    Location,
)

from ...support.logging import (
    runtime_logger as logger,
)

from ..device import (
    Device,
)

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


# TODO: The cache should be more than just a simple dict. Can be persistent
KERNEL_CACHE: dict[str, tuple[VmContext, VmFunction, KernelCompileConfig]] = {}


def _testing_get_cache_size() -> int:
    return len(KERNEL_CACHE)


def compile_standalone_kernel(
    device: Device, ksel: KernelSelection, func_name: str = "main"
) -> tuple[VmContext, VmFunction, KernelCompileConfig]:
    # Early exit on cache hit.
    cache_key = f"{ksel.spec_key}::{device.type_cache_key}"
    cache_hit = KERNEL_CACHE.get(cache_key)
    if cache_hit is not None:
        return cache_hit

    # Cache miss.
    start = default_timer()
    config = KernelCompileConfig(cache_key, list(device.compile_target_flags))
    kb = FreeFuncKernelBuilder.create_module(ksel, func_name=func_name)
    with kb.ip, Location.unknown():
        ksel.op.generate(ksel, kb)
    kb.module_op.verify()
    module_asm = kb.module_op.get_asm(
        binary=True, enable_debug_info=True, print_generic_op_form=True
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
    vm_module = VmModule.copy_buffer(vm_instance, mapped_memory)
    # TODO: We should be able to wrap the buffer as below but there are some
    # subtle ref-counting/shutdown sequencing issues that need to be resolved.
    # vm_module = VmModule.wrap_buffer(vm_instance, mapped_memory)
    vm_context = VmContext(vm_instance, [device.create_hal_module(), vm_module])
    main_function = vm_module.lookup_function("main")

    logger.debug(
        "Compiled kernel %s: mlir=%d bytes, vmfb=%d bytes (generation: %sms, compilation: %sms)",
        cache_key,
        len(module_asm),
        len(mapped_memory),
        generation_time * 1000,
        compilation_time * 1000,
    )
    cache_hit = (vm_context, main_function, config)
    KERNEL_CACHE[cache_key] = cache_hit
    return cache_hit
