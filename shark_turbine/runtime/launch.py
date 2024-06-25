# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from iree.compiler.api import (
    Session,
    Source,
    Output,
)

from iree.runtime import (
    create_io_parameters_module,
    HalBufferView,
    HalElementType,
    ParameterProvider,
    VmContext,
    VmFunction,
    VmModule,
    VmRef,
    VmVariantList,
)

from ..support.logging import runtime_logger as logger

from .device import (
    get_device_from_torch,
    Device,
)

__all__ = [
    "Launchable",
]

_TargetBinary = Tuple[VmContext, VmFunction]
_Loader = Callable[[Device], _TargetBinary]


class Launchable:
    """Facilities for launching a compiled program (VMFB) on an attached device.

    Like the eager custom-op executor, this follows the usual PyTorch rules
    whereby the device that input tensors reside on dictates where the launch
    happens. Unlike that flow, this does not include any notion of jitting
    or caching. It also has APIs for using parameters, etc.

    You must manage all compilation/target settings yourself and you merely
    assert that a given binary is appropriate for launch on a device type.
    This has various limitations.
    """

    def __init__(self, loader: Optional[_Loader]):
        self._loader = loader
        # Map of Device.type_cache_key -> _TargetBinary for a resolved binary.
        self._target_binaries: dict[str, _TargetBinary] = {}

    @staticmethod
    def jit_compile(
        source: Any,
        *,
        parameter_providers: Sequence[ParameterProvider] = (),
        entry_point: str = "main",
    ) -> "Launchable":
        return Launchable.from_vm_module(
            _jit_callback(source),
            parameter_providers=parameter_providers,
            entry_point=entry_point,
        )

    @staticmethod
    def from_vm_module(
        vm_module_callback: Callable[[Device], VmModule],
        *,
        parameter_providers: Sequence[ParameterProvider] = (),
        entry_point: str = "main",
    ) -> "Launchable":
        def loader(device: Device) -> _TargetBinary:
            vm_instance = device.vm_instance
            modules = [device.create_hal_module()]
            if parameter_providers:
                modules.append(
                    create_io_parameters_module(vm_instance, *parameter_providers)
                )
            main_module = vm_module_callback(device)
            modules.append(main_module)
            vm_context = VmContext(vm_instance, modules)
            main_function = main_module.lookup_function(entry_point)
            return vm_context, main_function

        return Launchable(loader)

    def preload(self, device: torch.device):
        """Pre-loads (or JIT compiles) for the given torch.device."""
        turbine_device = get_device_from_torch(device)
        self._resolve_target_binary(turbine_device)

    def _resolve_target_binary(self, turbine_device: Device) -> _TargetBinary:
        device_key = turbine_device.type_cache_key
        existing = self._target_binaries.get(device_key)
        if existing is not None:
            logger.debug("Launching cached binary for %s", device_key)
            return existing

        # Try the user loader.
        loader = self._loader
        if loader is not None:
            loaded = loader(turbine_device)
            if loaded is not None:
                logger.debug("Cached new binary for %s", device_key)
                self._target_binaries[device_key] = loaded
                return loaded
        raise NotImplementedError(
            f"Could not load a target binary for device {turbine_device}"
        )

    def __call__(self, *args, device: Optional[torch.device] = None):
        turbine_device: Optional[Device] = (
            None if device is None else get_device_from_torch(device)
        )
        arg_list = VmVariantList(len(args))
        # Scan args for tensors and infer device.
        for arg in args:
            if isinstance(arg, Tensor):
                # For pre-compiled launchables, there is no support for anything
                # but contiguous layouts.
                if not arg.is_contiguous():
                    arg = arg.contiguous()
                tensor_device = arg.device
                if device is None:
                    device = tensor_device
                else:
                    if tensor_device != device:
                        raise RuntimeError(
                            f"Cannot launch with tensors from multiple devices: "
                            f"{tensor_device} vs {device}"
                        )
                if turbine_device is None:
                    turbine_device = get_device_from_torch(tensor_device)
                # Since we know we are on the same device, we can use the unsafe
                # import_torch_tensor.
                arg_list.push_ref(turbine_device.import_torch_tensor(arg))
            elif isinstance(arg, int):
                arg_list.push_int(arg)
            elif isinstance(arg, float):
                arg_list.push_float(arg)

        # Having at least one tensor arg is a pre-requisite for normal operation
        if device is None or turbine_device is None:
            raise RuntimeError(
                f"Cannot invoke Launchable {self} without any Tensor args or an explicit device="
            )

        vm_context, vm_function = self._resolve_target_binary(turbine_device)
        ret_list = VmVariantList(1)
        vm_context.invoke(vm_function, arg_list, ret_list)
        torch_results = []
        for i in range(len(ret_list)):
            result = ret_list.get_variant(i)
            if isinstance(result, VmRef):
                buffer_view = result.deref(HalBufferView, True)
                if buffer_view is not None:
                    torch_results.append(
                        _export_torch_tensor(buffer_view, turbine_device)
                    )

        arity = len(torch_results)
        if arity == 1:
            return torch_results[0]
        elif arity == 0:
            return None
        else:
            return torch_results


def _jit_callback(program_source: Any) -> Callable[[Device], VmModule]:
    session = Session()
    if isinstance(program_source, Source):
        ...
    elif isinstance(program_source, str):
        source = Source.wrap_buffer(session, program_source.encode())
    else:
        source = Source.wrap_buffer(session, program_source)

    def callback(device: Device):
        session.set_flags(*device.compile_target_flags)
        inv = session.invocation()
        output = Output.open_membuffer()
        inv.enable_console_diagnostics()
        inv.parse_source(source)
        if not inv.execute():
            # TODO: Capture diagnostics and report.
            raise RuntimeError(f"JIT compilation failed. See diagnostics.")
        inv.output_vm_bytecode(output)
        mapped_memory = output.map_memory()
        vm_instance = device.vm_instance
        # TODO: VmModule.wrap_buffer would be better here, but it is still
        # unreliable capturing mapped memory from the compiler.
        # See: https://github.com/iree-org/iree/issues/17403
        return VmModule.copy_buffer(vm_instance, mapped_memory)

    return callback


def _export_torch_tensor(bv: HalBufferView, turbine_device: Device) -> Tensor:
    # Usually in the custom op flow, we have strong metadata about the results.
    # But since the whole purpose of this is for interfacing a blackbox, we
    # just infer from IREE type -> torch type. This may be lossy on dtypes
    # that are not an exact match, and the user is expected to bitcast.
    dtype = _INFERRED_ELEMENT_TYPE_TO_DTYPE[int(bv.element_type)]
    if dtype is None:
        raise NotImplementedError(
            f"HalBufferView.element_type({bv.element_type}) has no mapping to dtype"
        )
    meta_tensor = torch.empty(bv.shape, dtype=dtype, device="meta")
    return turbine_device.export_torch_tensor(bv, meta_tensor)


# This is a relatively special purpose mapping. We usually don't go this
# way because it is lossy: IREE's types are "fundamental" and lacking
# signed/unsigned distinctions at this layer, so we do the best we can.
# If this becomes less special purpose, move it to conversions.py
_INFERRED_ELEMENT_TYPE_TO_DTYPE: dict[HalElementType, torch.dtype] = {
    int(HalElementType.BFLOAT_16): torch.bfloat16,
    int(HalElementType.BOOL_8): torch.bool,
    int(HalElementType.COMPLEX_64): torch.complex64,
    int(HalElementType.COMPLEX_128): torch.complex128,
    int(HalElementType.FLOAT_16): torch.float16,
    int(HalElementType.FLOAT_32): torch.float32,
    int(HalElementType.FLOAT_64): torch.float64,
    int(HalElementType.INT_8): torch.int8,
    int(HalElementType.INT_16): torch.int16,
    int(HalElementType.INT_32): torch.int32,
    int(HalElementType.INT_64): torch.int64,
    int(HalElementType.SINT_8): torch.int8,
    int(HalElementType.SINT_16): torch.int16,
    int(HalElementType.SINT_32): torch.int32,
    int(HalElementType.SINT_64): torch.int64,
    int(HalElementType.UINT_8): torch.uint8,
    int(HalElementType.UINT_16): torch.uint16,
    int(HalElementType.UINT_32): torch.uint32,
    int(HalElementType.UINT_64): torch.uint64,
}
