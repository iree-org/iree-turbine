# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import lru_cache
from typing import Callable, Optional, Union
from threading import local, Lock

import torch

from iree.runtime import (
    BufferUsage,
    HalBufferView,
    HalDevice,
    HalDriver,
    MemoryType,
    VmInstance,
    VmModule,
    create_hal_module,
    get_driver,
)

from ..support.conversions import (
    dtype_to_element_type,
    torch_dtype_to_numpy,
)

from ..support.exceptions import (
    NoCurrentDeviceError,
    MismatchedDeviceSetClearError,
    UnsupportedTorchDeviceError,
)

from ..support.logging import runtime_logger as logger

__all__ = [
    "get_vm_instance",
    "Device",
    "DeviceState",
]

_CONFIG_LOCK = Lock()
_GLOBAL_VM_INSTANCE: Optional[VmInstance] = None
_CURRENT_THREAD = local()

###############################################################################
# DeviceState ande Device classes.
# These associated shared VmInstance and HalDrivers with a concrete HalDevice.
# The Device class also adds other accounting needed for interop in PyTorch's
# eager environment (i.e. transfer and compute queue counters, etc).
###############################################################################


def get_vm_instance() -> VmInstance:
    global _GLOBAL_VM_INSTANCE
    if not _GLOBAL_VM_INSTANCE:
        with _CONFIG_LOCK:
            if not _GLOBAL_VM_INSTANCE:
                _GLOBAL_VM_INSTANCE = VmInstance()
    return _GLOBAL_VM_INSTANCE


class DeviceState:
    """State for an instantiated HAL device.

    Note that the IREE runtime internally manages a global cache of drivers for
    standard named-access (not custom-constructed) drivers.
    """

    __slots__ = [
        "device",
        "driver",
        "instance",
    ]

    def __init__(
        self,
        *,
        driver: Union[str, HalDriver],
        device: Optional[HalDevice] = None,
        vm_instance: Optional[VmInstance] = None,
    ):
        self.instance = vm_instance or get_vm_instance()
        self.driver = driver if isinstance(driver, HalDriver) else get_driver(driver)
        self.device = device if device else self.driver.create_default_device()

    @staticmethod
    @lru_cache(maxsize=None)
    def from_uri(uri: str) -> "DeviceState":
        driver = get_driver(uri)
        return DeviceState(driver=driver, device=driver.create_device_by_uri(uri))


class Device:
    """Represents a low-level device (HalDriver/HalDevice) and scheduling data.

    This is the type that user's interact with as a 'Device'. Devices can be handled
    loose-leaf or bound to a thread with a context manager.
    """

    __slots__ = [
        "_s",
        "_main_timeline",
        "_main_timepoint",
        "_tx_timeline",
        "_tx_timepoint",
        "_fence_capacity",
        "compile_target_flags",
        "export_torch_tensor",
        "import_torch_tensor",
        "instance_cache_key",
        "type_cache_key",
    ]

    _s: DeviceState

    # Each device will have a function attached to import a torch.tensor
    # *that is already on that device* directly from device memory.
    # This is unsafe and relatively unchecked. If criss-crossing devices,
    # it is undefined behavior.
    import_torch_tensor: Callable[[torch.Tensor], HalBufferView]

    # Devices can also export a torch tensor from a HalBufferView, given
    # a meta tensor that describes it.
    export_torch_tensor: Callable[[HalBufferView, torch.Tensor], torch.Tensor]

    # Cache key that uniquely identifies this device.
    instance_cache_key: str

    # Cache key that uniquely identifies this type of device (currently
    # based on its driver).
    type_cache_key: str

    # Compiler flags to use to target this device.
    # TODO: We should replace this with a target attribute but need an API
    # to derive that.
    compile_target_flags: tuple[str, ...]

    def __new__(
        cls, uri: Optional[str] = None, *, device_state: Optional[DeviceState] = None
    ):
        if uri is not None:
            # Construction by URI is cached on the thread.
            assert not device_state, "device_state= cannot be given with explicit URI"
            try:
                existing = _CURRENT_THREAD.device_by_uri[uri]
            except (AttributeError, KeyError):
                ...
            else:
                return existing

            # New instance.
            device_state = DeviceState.from_uri(uri)
            new_inst = super().__new__(cls)
            new_inst._s = device_state
            try:
                _CURRENT_THREAD.device_by_uri[uri] = new_inst
            except AttributeError:
                _CURRENT_THREAD.device_by_uri = {uri: new_inst}
            new_inst._initialize()
            return new_inst
        else:
            # Explicit construction with a device_state is assumed that you know what you
            # are doing and an uncached instance will be returned. This will be unsychronized
            # relative to any cached instance.
            assert device_state, "device_state= must be given if URI ommitted"
            new_inst = super().__new__(cls)
            new_inst._s = device_state
            new_inst._initialize()
            return new_inst

    def _initialize(self):
        d = self._s.device
        self._main_timeline = d.create_semaphore(0)
        self._main_timepoint = 0
        self._tx_timeline = d.create_semaphore(0)
        self._tx_timepoint = 0
        # Maximum number of semaphores the device uses. Can be increased if doing out of the
        # ordinary scheduling.
        self._fence_capacity = 2

        # Perform driver specific augmentations.
        # TODO: Add a HalDriver.id property to get the driver name instead of parsing
        # the device repr.
        driver_id = repr(d)
        colon_pos = driver_id.find(":")
        if colon_pos >= 0:
            driver_id = driver_id[0:colon_pos]
        try:
            import_fn = TORCH_TENSOR_IMPORTERS[driver_id]
            export_fn = TORCH_TENSOR_EXPORTERS[driver_id]
            self.import_torch_tensor = lambda t: import_fn(self, t)
            self.export_torch_tensor = lambda bv, t: export_fn(self, bv, t)
            self.compile_target_flags = DEVICE_TARGET_COMPILE_FLAGS[driver_id]
        except KeyError as e:
            raise AssertionError(
                f"Unsupported TORCH_TENSOR_IMPORTERS for iree driver '{driver_id}'"
            ) from e

        # Cache keys.
        # TODO: The type cache key should actually be based on the driver id
        # and device characteristics hash.
        self.instance_cache_key = repr(d)
        self.type_cache_key = driver_id

    @property
    def hal_device(self) -> HalDevice:
        return self._s.device

    @property
    def vm_instance(self) -> VmInstance:
        return self._s.instance

    def create_hal_module(self) -> VmModule:
        s = self._s
        return create_hal_module(s.instance, s.device)

    def current() -> "Device":
        try:
            return _CURRENT_THREAD.stack[-1]
        except (AttributeError, IndexError):
            raise NoCurrentDeviceError()

    def set(self) -> "Device":
        """Sets this device as the current device without a context manager."""
        try:
            _CURRENT_THREAD.stack.append(self)
        except AttributeError:
            _CURRENT_THREAD.stack = [self]

    def clear(self):
        """Clears the current device without a context manager."""
        try:
            c = _CURRENT_THREAD.stack[-1]
            if _CURRENT_THREAD.stack[-1] is self:
                _CURRENT_THREAD.stack.pop()
                return
        except (AttributeError, IndexError):
            ...
        raise MismatchedDeviceSetClearError()

    def __repr__(self):
        return f"<Turbine Device: {self._s.device}>"

    def __enter__(self):
        try:
            _CURRENT_THREAD.stack.append(self)
        except AttributeError:
            _CURRENT_THREAD.stack = [self]

    def __exit__(self, type, value, traceback):
        _CURRENT_THREAD.stack.pop()


def _device_import_torch_tensor_cpu(device: Device, t: torch.Tensor) -> HalBufferView:
    hal_device = device.hal_device
    element_type = dtype_to_element_type(t.dtype)
    # TODO: In this case, we should be importing the raw buffer, but this is not
    # generically exposed to Python in the IREE runtime.
    bv = device.hal_device.allocator.allocate_buffer_copy(
        memory_type=MemoryType.DEVICE_LOCAL,
        allowed_usage=BufferUsage.DEFAULT,
        device=hal_device,
        buffer=t.detach().numpy(),
        element_type=element_type,
    )
    return bv


def _device_export_torch_tensor_cpu(
    device: Device, bv: HalBufferView, like: torch.Tensor
) -> torch.Tensor:
    # TODO: Similar to import, we know that the buffer is in local CPU memory
    # and could export it if we had Python API support for that. Until we have
    # that, we do this very torturous indirection.
    mapped_memory = bv.map()
    shape = list(like.shape)
    np_dtype = torch_dtype_to_numpy(like.dtype)
    mapped_array = mapped_memory.asarray(shape, np_dtype)
    return torch.from_numpy(mapped_array)


# Mapping of torch tensor importers keyed by driver name.
TORCH_TENSOR_IMPORTERS: dict[str, Callable[[Device, torch.Tensor], HalBufferView]] = {
    "local-sync": _device_import_torch_tensor_cpu,
    "local-task": _device_import_torch_tensor_cpu,
}

TORCH_TENSOR_EXPORTERS: dict[
    str, Callable[[Device, HalBufferView, torch.Tensor], torch.Tensor]
] = {
    "local-sync": _device_export_torch_tensor_cpu,
    "local-task": _device_export_torch_tensor_cpu,
}

DEVICE_TARGET_COMPILE_FLAGS: dict[str, tuple[str, ...]] = {
    "local-task": ("--iree-hal-target-backends=llvm-cpu",),
}

# Aliases.
DEVICE_TARGET_COMPILE_FLAGS["local-sync"] = DEVICE_TARGET_COMPILE_FLAGS["local-task"]

# Make sure all tables have the same keys.
assert (
    TORCH_TENSOR_IMPORTERS.keys() == DEVICE_TARGET_COMPILE_FLAGS.keys()
), "Not all devices have the same configs"

assert (
    TORCH_TENSOR_IMPORTERS.keys() == TORCH_TENSOR_EXPORTERS.keys()
), "Not all devices have the same configs"

###############################################################################
# torch.device to Device mapping
###############################################################################


def lookup_device_from_torch(
    torch_device: torch.device, *, create: bool = True
) -> Optional[Device]:
    """Gets a shared Device corresponding to the given torch.device.

    This will return None if the device is wholly unsupported or if
    create=False. Otherwise, faults in setting up the device are
    reported as an appropriate exception.
    """
    try:
        mapping = _CURRENT_THREAD.device_by_torch_device
    except AttributeError:
        _CURRENT_THREAD.device_by_torch_device = mapping = {}
    device = mapping.get(torch_device)
    if device is not None or not create:
        return device
    logger.debug("Creating turbine device for torch.device = %r", torch_device)
    device = _create_device_from_torch(torch_device)
    if device is not None:
        mapping[torch_device] = device
    return device


def get_device_from_torch(torch_device: torch.device) -> Device:
    """Gets a shared Device corresponding to the given torch.device.

    Raises an exception if the device cannot be created.
    """
    device = lookup_device_from_torch(torch_device)
    if device is None:
        raise UnsupportedTorchDeviceError(torch_device)
    return device


def _create_device_from_torch(torch_device: torch.device) -> Optional[Device]:
    torch_type = torch_device.type
    uri = None
    if torch_type == "cpu":
        uri = "local-task"

    if uri is None:
        return None

    return Device(uri)


###############################################################################
# Utilities
###############################################################################

# The nanobind leak checker doesn't interop well with the way that
# global state is managed for PyTorch. It isn't clear that this
# is a fully correctable state of affairs, so we just disable it
# for now. RIP nice things :(
from iree.runtime._binding import disable_leak_checker

disable_leak_checker()
