# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from functools import lru_cache
from hashlib import sha1
from typing import Any, Callable, Dict, Optional, Union
from threading import local, Lock

import warnings
import platform
import atexit

import ctypes
import torch

from iree.runtime import (
    BufferUsage,
    HalBufferView,
    HalDevice,
    HalDriver,
    HalExternalTimepoint,
    MemoryType,
    VmInstance,
    VmModule,
    SemaphoreCompatibility,
    ExternalTimepointFlags,
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


# TODO: move this down into iree as an extention to the
#       driver api.
class _HipSemaphoreInterop:
    def __init__(self, sync):
        if platform.system() == "Windows":
            self.library = ctypes.CDLL("amdhip64.dll")
        else:
            self.library = ctypes.CDLL("libamdhip64.so")
        self.library.hipEventCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self.library.hipEventCreate.restype = ctypes.c_int32

        self.library.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.hipEventRecord.restype = ctypes.c_int32

        self.library.hipEventDestroy.argtypes = [ctypes.c_void_p]
        self.library.hipEventDestroy.restype = ctypes.c_int32

        self.library.hipStreamWaitEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.library.hipStreamWaitEvent.restype = ctypes.c_int32

        self.library.hipEventQuery.argtypes = [ctypes.c_void_p]
        self.library.hipEventQuery.restype = ctypes.c_int32
        self.sync = sync

    def get_timepoint_import(self):
        if not self.sync:
            return None
        evt = ctypes.c_void_p(0)
        ret = self.library.hipEventCreate(evt, 2)
        if ret != 0:
            raise RuntimeError("Could not create hip event")
        ret = self.library.hipEventRecord(
            evt, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        )
        if ret != 0:
            raise RuntimeError("Could not record hip event")
        timepoint = HalExternalTimepoint()
        timepoint.compatibility = SemaphoreCompatibility.DEVICE_WAIT
        timepoint.flags = ExternalTimepointFlags.NONE
        timepoint.hip_event = evt.value
        return timepoint

    def wait_exported_timepoint(self, timepoint: HalExternalTimepoint):
        if not self.sync:
            return
        ret = self.library.hipStreamWaitEvent(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
            ctypes.c_void_p(timepoint.hip_event),
            0,
        )
        if ret != 0:
            raise RuntimeError("Could not wait on event")

    def destroy_timepoint_event(self, timepoint: HalExternalTimepoint):
        if not self.sync:
            return True
        ret = self.library.hipEventDestroy(ctypes.c_void_p(timepoint.hip_event))
        if ret != 0:
            raise RuntimeError(f"Could not destroy event got {ret}")
        return True


class _CudaSemaphoreInterop:
    def __init__(self, sync):
        self.sync = sync
        pass

    def get_timepoint_import(self):
        # For now we don't actually support timepoint import in Cuda.
        # So we fall back to the synchronous approach.
        torch.cuda.current_stream().synchronize()
        return None

    def wait_exported_timepoint(self, timepoint: HalExternalTimepoint):
        pass

    def destroy_timepoint_event(self, timepoint: HalExternalTimepoint):
        return True


class _NullSemaphoreInterop:
    def __init__(self, sync):
        self.sync = sync

    def get_timepoint_import(self):
        return None

    def wait_exported_timepoint(self, timepoint: HalExternalTimepoint):
        pass

    def destroy_timepoint_event(self, timepoint: HalExternalTimepoint):
        return True


_CONFIG_LOCK = Lock()
_GLOBAL_VM_INSTANCE: Optional[VmInstance] = None
_CURRENT_THREAD = local()

###############################################################################
# DeviceState ande Device classes.
# These associated shared VmInstance and HalDrivers with a concrete HalDevice.
# The Device class also adds other accounting needed for interop in PyTorch's
# eager environment (i.e. transfer and compute queue counters, etc).
###############################################################################


DEFAULT_KEY_HASHER: Callable[[str], str] = lambda key: sha1(
    key.encode(), usedforsecurity=False
).hexdigest()


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
        "enumerated_info",
        "torch_device",
        "torch_stream",
        "dlpack_device_type_code",
    ]

    def __init__(
        self,
        *,
        driver: Union[str, HalDriver],
        device: Optional[HalDevice] = None,
        vm_instance: Optional[VmInstance] = None,
        enumerated_info: Optional[dict] = None,
        torch_device: Optional[torch.device] = None,
        torch_stream: Optional[int] = None,
        dlpack_device_type_code: int = 0,
    ):
        self.instance = vm_instance or get_vm_instance()
        self.driver = driver if isinstance(driver, HalDriver) else get_driver(driver)
        self.device = device if device else self.driver.create_default_device()
        self.enumerated_info = enumerated_info or {}
        self.torch_device = torch_device
        self.torch_stream = torch_stream
        self.dlpack_device_type_code = dlpack_device_type_code

    @property
    def enumerated_device_id(self) -> int:
        try:
            return self.enumerated_info["device_id"]
        except KeyError as e:
            raise RuntimeError("No enumerated device_id for device") from e

    @property
    def enumerated_path(self) -> str:
        try:
            return self.enumerated_info["path"]
        except KeyError as e:
            raise RuntimeError("No enumerated path for device") from e

    @property
    def enumerated_name(self) -> str:
        try:
            return self.enumerated_info["name"]
        except KeyError as e:
            raise RuntimeError("No enumerated name for device") from e

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
        "_external_timepoints",
        "_device_interop",
        "compile_target_flags",
        "driver_id",
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

    # Unique name of the IREE runtime driver associated with this device.
    driver_id: str

    # Cache key that uniquely identifies this device.
    instance_cache_key: str

    # Cache key that uniquely identifies this type of device (currently
    # based on its driver).
    type_cache_key: str

    # Compiler flags to use to target this device.
    # TODO: We should replace this with a target attribute but need an API
    # to derive that.
    compile_target_flags: tuple[str, ...]

    def _try_clean_external_timepoints(self):
        while len(self._external_timepoints) > 0:
            if self._main_timeline.query() >= self._external_timepoints[0][1]:
                self._device_interop.destroy_timepoint_event(
                    self._external_timepoints[0][0]
                )
                self._external_timepoints = self._external_timepoints[1:]
            else:
                break

    def setup_iree_action(self):
        self._try_clean_external_timepoints()
        timepoint_import = self._device_interop.get_timepoint_import()
        if timepoint_import is not None:
            self._main_timepoint += 1
            self._main_timeline.import_timepoint(self._main_timepoint, timepoint_import)
            timepoint_export = HalExternalTimepoint()
            self._main_timepoint += 1
            self._main_timeline.export_timepoint(
                self._main_timepoint,
                3,  # ExternalTimepointType.HIP_EVENT
                0,  # ExternalTimepointFlags.NONE,
                timepoint_export,
            )
            return timepoint_export
        elif self.sync:
            self._main_timepoint += 1
            return None

    def finalize_iree_action(self, external_timepoint: HalExternalTimepoint):
        if external_timepoint is not None:
            self._try_clean_external_timepoints()
            self._device_interop.wait_exported_timepoint(external_timepoint)
            self._external_timepoints.append((external_timepoint, self._main_timepoint))
        elif self.sync:
            self._main_timeline.wait(self._main_timepoint)

    def __new__(
        cls,
        uri: Optional[str] = None,
        *,
        device_state: Optional[DeviceState] = None,
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
        self._external_timepoints = []
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
        self.driver_id = driver_id
        try:
            import_fn = TORCH_TENSOR_IMPORTERS[driver_id]
            export_fn = TORCH_TENSOR_EXPORTERS[driver_id]
            self._device_interop = IREE_SEMAPHPORE_INTEROP[driver_id](
                self._s.torch_stream is None
            )
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
        self._recompute_target_keys()

        # This is a bit unfortunate, but our external timepoints
        #  are ephemeral, so we need to hold onto them after
        #  any calls into the device (therefore we have nowhere)
        #  clean to destroy them. So make sure we destroy
        #  any remaining external timepoints before the application
        #  closes.
        atexit.register(self._try_clean_external_timepoints)

    def _recompute_target_keys(self):
        self.type_cache_key = f"{self.driver_id}:{';'.join(self.compile_target_flags)}"
        self.instance_cache_key = f"{self.driver_id}:{self._s.enumerated_info.get('device_id', None)}:{self._s.torch_stream}"

    @property
    def hal_device(self) -> HalDevice:
        return self._s.device

    @property
    def vm_instance(self) -> VmInstance:
        return self._s.instance

    @property
    def sync(self) -> bool:
        return self._s.torch_stream is None

    def create_hal_module(self) -> VmModule:
        s = self._s
        return create_hal_module(s.instance, s.device)

    def get_type_key_hash(self, *, hasher: Callable[[str], str] = DEFAULT_KEY_HASHER):
        return hasher(self.type_cache_key)

    @staticmethod
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
        return self

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

    def dump_device_info(self) -> str:
        return self._s.driver.dump_device_info(self._s.enumerated_device_id)

    def __repr__(self):
        return f"<Turbine Device: {self._s.device}>"

    def __enter__(self):
        try:
            _CURRENT_THREAD.stack.append(self)
        except AttributeError:
            _CURRENT_THREAD.stack = [self]

    def __exit__(self, type, value, traceback):
        _CURRENT_THREAD.stack.pop()


################################################################################
# CPU import/export
################################################################################


def _device_import_torch_tensor_cpu(device: Device, t: torch.Tensor) -> HalBufferView:
    hal_device = device.hal_device
    element_type = dtype_to_element_type(t.dtype)

    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)

    if t.dtype in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }:
        t = t.view(torch.int8)

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
    return torch.from_numpy(mapped_array).view(like.dtype)


################################################################################
# CUDA and HIP import/export
################################################################################


def _device_import_torch_tensor_cuda_hip(
    device: Device, t: torch.Tensor
) -> HalBufferView:
    # We currently only support contiguous, so ensure that.
    if not t.is_contiguous():
        t = t.contiguous()
    # The None passed to tensor.__dlpack__ indicates we are doing no stream synchronization here.
    # We launch kernels through IREE runtime on the same stream as pytorch. If using multiple
    # streams, the user is expected to properly manage stream synchronization.
    capsule = t.__dlpack__(None)
    bv = device.hal_device.from_dlpack_capsule(capsule)
    return bv


def _device_export_torch_tensor_cuda_hip(
    device: Device, bv: HalBufferView, like: torch.Tensor
) -> torch.Tensor:
    state = device._s
    device_type_code = state.dlpack_device_type_code
    assert device_type_code > 0
    torch_device = state.torch_device
    assert torch_device is not None
    device_index = torch_device.index
    t = torch.from_dlpack(
        device.hal_device.create_dlpack_capsule(bv, device_type_code, device_index)
    )
    if t.dtype != like.dtype:
        t = t.view(like.dtype)
    return t


# Mapping of torch tensor importers keyed by driver name.
TORCH_TENSOR_IMPORTERS: dict[str, Callable[[Device, torch.Tensor], HalBufferView]] = {
    "cuda": _device_import_torch_tensor_cuda_hip,
    "hip": _device_import_torch_tensor_cuda_hip,
    "local-sync": _device_import_torch_tensor_cpu,
    "local-task": _device_import_torch_tensor_cpu,
}

TORCH_TENSOR_EXPORTERS: dict[
    str, Callable[[Device, HalBufferView, torch.Tensor], torch.Tensor]
] = {
    "cuda": _device_export_torch_tensor_cuda_hip,
    "hip": _device_export_torch_tensor_cuda_hip,
    "local-sync": _device_export_torch_tensor_cpu,
    "local-task": _device_export_torch_tensor_cpu,
}

IREE_SEMAPHPORE_INTEROP: dict[str, type] = {
    "cuda": _CudaSemaphoreInterop,
    "hip": _HipSemaphoreInterop,
    "local-sync": _NullSemaphoreInterop,
    "local-task": _NullSemaphoreInterop,
}

DEVICE_TARGET_COMPILE_FLAGS: dict[str, tuple[str, ...]] = {
    "cuda": ("--iree-hal-target-backends=cuda",),
    "hip": ("--iree-hal-target-backends=rocm",),
    "local-task": (
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu-features=host",
    ),
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
    stream = (
        None
        if torch_device.type != "cuda"
        else torch.cuda.current_stream(torch_device).cuda_stream
    )
    key = (torch_device, stream)
    device = mapping.get(key)
    if device is not None or not create:
        return device
    logger.debug("Creating turbine device for torch.device = %r", torch_device)
    device = _create_device_from_torch(torch_device)
    if device is not None:
        mapping[key] = device
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
    if torch_type == "cpu":
        cpu_driver = get_driver("local-task")
        cpu_enumerated = cpu_driver.query_available_devices()
        assert len(cpu_enumerated) >= 1
        cpu_default = cpu_enumerated[0]
        cpu_device_state = DeviceState(
            driver=cpu_driver,
            device=cpu_driver.create_default_device(),
            enumerated_info=cpu_default,
            torch_device=torch_device,
            dlpack_device_type_code=1,
        )
        return Device(device_state=cpu_device_state)
    elif torch_type == "cuda":
        # Fork based on HIP or real CUDA.
        props = torch.cuda.get_device_properties(torch_device)
        if not hasattr(props, "gcnArchName"):
            # Real CUDA.
            return _create_cuda_device(torch_device, props)
        else:
            # HIP as CUDA.
            return _create_hip_device(torch_device, props)

    return None


def _create_cuda_device(torch_device: torch.device, props) -> Optional[Device]:
    # Note that the dlpack device type code for real CUDA ROCM is 2.
    device = _create_cuda_like_device(torch_device, props, "hip", 2, None)
    if device:
        device.compile_target_flags = device.compile_target_flags + (
            f"--iree-hal-cuda-llvm-target-arch=sm_{props.major}{props.minor}",
        )
        device._recompute_target_keys()
    return device


def _create_hip_device(torch_device: torch.device, props) -> Optional[Device]:
    stream = torch.cuda.current_stream(torch_device).cuda_stream
    device_params = {"hip_external_stream": str(stream)}
    # Note that the dlpack device type code for ROCM is 10.
    device = _create_cuda_like_device(torch_device, props, "hip", 10, device_params)
    # The gcnArchName comes back like gfx90a:sramecc+:xnack- for a fully
    # specified target. However the IREE target-chip flag only expects the
    # prefix. See: https://github.com/iree-org/iree/issues/17402
    # This should be changed to tunnel through target information unmolested.
    gcn_arch_name: str = props.gcnArchName
    colon_pos = gcn_arch_name.find(":")
    if colon_pos >= 0:
        gcn_arch_name = gcn_arch_name[0:colon_pos]
    if device:
        gcn_arch_name = gcn_arch_name
        device.compile_target_flags = device.compile_target_flags + (
            f"--iree-hip-target={gcn_arch_name}",
        )
        device._recompute_target_keys()
    return device


@lru_cache(maxsize=None)
def _get_uuid_to_info_mapping(driver) -> Dict[str, Dict[str, Any]]:
    available_infos = driver.query_available_devices()
    return {info["path"].removeprefix("GPU-"): info for info in available_infos}


def _create_cuda_like_device(
    torch_device: torch.device,
    props,
    driver_name: str,
    dlpack_device_type_code: int,
    device_params,
) -> Optional[Device]:
    uuid = str(torch.cuda.get_device_properties(torch_device).uuid)
    driver = get_driver(driver_name)
    info_mapping = _get_uuid_to_info_mapping(driver)
    device_info = info_mapping.get(uuid)
    if device_info is None:
        return None
    hal_device = driver.create_device(device_info, device_params)
    device_state = DeviceState(
        driver=driver,
        device=hal_device,
        vm_instance=get_vm_instance(),
        enumerated_info=device_info,
        torch_device=torch_device,
        torch_stream=torch.cuda.current_stream(torch_device).cuda_stream,
        dlpack_device_type_code=dlpack_device_type_code,
    )
    device = Device(device_state=device_state)
    return device


###############################################################################
# Utilities
###############################################################################

# The nanobind leak checker doesn't interop well with the way that
# global state is managed for PyTorch. It isn't clear that this
# is a fully correctable state of affairs, so we just disable it
# for now. RIP nice things :(
from iree.runtime._binding import disable_leak_checker

disable_leak_checker()
