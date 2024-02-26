# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import os
from typing import List, Optional, Sequence, Union
from dataclasses import dataclass
from iree.runtime import (
    asdevicearray,
    create_hal_module,
    HalBuffer,
    HalBufferView,
    HalFence,
    HalElementType,
    DeviceArray,
    get_driver,
    VmContext,
    HalDevice,
    HalDriver,
    VmInstance,
    VmModule,
    VmVariantList,
)

import torch
from torch import (
    from_numpy as torch_from_numpy,
)

from ..runtime.device import Device, DeviceState


@functools.lru_cache(maxsize=None)
def get_vm_instance() -> VmInstance:
    return VmInstance()


_ELEMENT_TYPE_TO_DTYPE = {
    HalElementType.FLOAT_16: torch.float16,
    HalElementType.BFLOAT_16: torch.bfloat16,
    HalElementType.FLOAT_32: torch.float32,
    HalElementType.FLOAT_64: torch.float64,
    HalElementType.UINT_8: torch.uint8,
    HalElementType.SINT_8: torch.int8,
    HalElementType.SINT_16: torch.int16,
    HalElementType.SINT_32: torch.int32,
    HalElementType.SINT_64: torch.int64,
    HalElementType.BOOL_8: torch.bool,
    HalElementType.OPAQUE_8: torch.qint8,
    HalElementType.OPAQUE_8: torch.quint8,
    HalElementType.COMPLEX_64: torch.complex64,
    HalElementType.COMPLEX_128: torch.complex128,
}


class SpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    __slots__ = [
        "device_state",
        "entry_function",
        "user_module",
        "vm_context",
    ]

    def __init__(
        self,
        user_module: VmModule,
        device_state: DeviceState,
        entry_name: str = "main",
    ):
        self.user_module = user_module
        self.vm_context = VmContext(
            device_state.instance,
            (
                create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list)
        # TODO: Append semaphores for async execution.

        # Invoke.
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list)

    def _inputs_to_device(self, inputs: list, arg_list: VmVariantList):
        # TODO: We are assuming the worst case here which is that we have unknown Torch
        # tensors that we send to the CPU and make continguous. Ideally, we would have
        # fast paths for our own backends and interop.
        for input in inputs:
            input_cpu = input.cpu().contiguous()
            # Since this is already a fallback case, just use the numpy array interop.
            # It isn't great, but meh... fallback case.
            device_array = asdevicearray(self.device_state.device, input_cpu)
            arg_list.push_ref(device_array._buffer_view)

    def _returns_to_user(self, ret_list: VmVariantList):
        # TODO: This is also not good that we are moving back to the CPU like this.
        # We should be returning a custom Tensor implementation which represents
        # our device data and has synchronization hooks for accessing it.
        device = self.device_state.device
        num_returns = len(ret_list)
        user_returns = [None] * num_returns
        for i in range(num_returns):
            device_buffer_view = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
            device_array = DeviceArray(device, device_buffer_view)
            host_array = device_array.to_host()
            user_returns[i] = torch_from_numpy(host_array)

        return user_returns


@dataclass
class EagerExecResult:
    buffer: HalBuffer
    size: int
    dtype: torch.dtype
    signal: Optional[HalFence] = None


def _element_type_to_dtype(element_type) -> torch.dtype:
    try:
        return _ELEMENT_TYPE_TO_DTYPE[element_type]
    except KeyError:
        raise ValueError(f"Unable to map {element_type} to torch dtype.")


class EagerSpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    __slots__ = [
        "device_state",
        "entry_function",
        "user_module",
        "vm_context",
    ]

    def __init__(
        self,
        user_module: VmModule,
        device_state: DeviceState,
        entry_name: str = "main$async",
    ):
        self.user_module = user_module
        self.vm_context = VmContext(
            device_state.instance,
            (
                create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        arg_list = VmVariantList(len(inputs))
        ret_list = VmVariantList(
            1
        )  # TODO: Get the number of results from the descriptor.

        # Initialize wait and signal fence if not async mode.
        device = inputs[0]._storage.device
        wait_fence, signal_fence = self._initialize_fences(device, inputs, arg_list)

        # Move inputs to the device and add to arguments.
        self._inputs_to_device(inputs, arg_list, wait_fence, signal_fence)

        # Invoke.
        self.vm_context.invoke(self.entry_function, arg_list, ret_list)
        return self._returns_to_user(ret_list, signal_fence)

    def _inputs_to_device(
        self,
        inputs: list,
        arg_list: VmVariantList,
        wait_fence: HalFence = None,
        signal_fence: HalFence = None,
    ):
        # TODO: We are assuming the worst case here which is that we have unknown Torch
        # tensors that we send to the CPU and make continguous. Ideally, we would have
        # fast paths for our own backends and interop.
        for input in inputs:
            arg_list.push_ref(input.buffer_view)
            wait_fence.extend(input._storage.ready_fence)

        # Append fences into list.
        arg_list.push_ref(wait_fence)
        arg_list.push_ref(signal_fence)

    def _returns_to_user(self, ret_list: VmVariantList, signal: HalFence = None):
        # TODO: This is also not good that we are moving back to the CPU like this.
        # We should be returning a custom Tensor implementation which represents
        # our device data and has synchronization hooks for accessing it.
        device = self.device_state.device
        num_returns = len(ret_list)
        user_returns = [None] * num_returns
        for i in range(num_returns):
            device_buffer_view = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
            dtype = _element_type_to_dtype(device_buffer_view.element_type)
            size = torch.Size(device_buffer_view.shape)
            device_buffer = device_buffer_view.get_buffer()
            user_returns[i] = EagerExecResult(device_buffer, size, dtype, signal)
        return user_returns

    def _initialize_fences(self, device: Device, inputs: list, arg_list: VmVariantList):
        fence_capacity = device._fence_capacity
        tx_semaphore = device._tx_timeline
        current_tx_timepoint = device._tx_timepoint

        # Create wait semaphore and fence.
        wait_semaphores = (tx_semaphore, current_tx_timepoint)
        wait_fence = HalFence(fence_capacity)
        wait_fence.insert(*wait_semaphores)

        # Create signal semaphore and fence.
        device._tx_timepoint += 1
        signals_semaphore = (tx_semaphore, current_tx_timepoint + 1)
        signal_fence = HalFence(fence_capacity)
        signal_fence.insert(*signals_semaphore)

        # Add fences into arg_list for async exec.
        return wait_fence, signal_fence
