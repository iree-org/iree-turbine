# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Any,
    Callable,
    Sequence,
)

from .device import Device

from iree.runtime import (
    VmContext,
    VmFunction,
    HalFence,
    VmVariantList,
)


__all__ = [
    "invoke_vm_function",
]


def invoke_vm_function(
    device: Device,
    is_async: bool,
    vm_context: VmContext,
    vm_function: VmFunction,
    arg_list: VmVariantList,
    ret_list: VmVariantList,
    *,
    timer: Callable[[], float] = (lambda: 0.0)
):
    """Invokes a vm function on a device, adding async fences to the arg_list if is_async.

    No checks are made to ensure compatibility between the provided device and vm_function.
    A timer function (float return) may be provided, and this function will return the invocation time.
    """

    if is_async:
        external_timepoint = device.setup_iree_action()
        if device.sync:
            wait_fence = HalFence.create_at(
                device._main_timeline, device._main_timepoint - 1
            )
            signal_fence = HalFence.create_at(
                device._main_timeline, device._main_timepoint
            )
        else:
            wait_fence = HalFence(0)
            signal_fence = HalFence(0)

        arg_list.push_ref(wait_fence)
        arg_list.push_ref(signal_fence)

    # Invoke.
    start = timer()
    vm_context.invoke(vm_function, arg_list, ret_list)

    if is_async:
        device.finalize_iree_action(external_timepoint)

    return timer() - start
