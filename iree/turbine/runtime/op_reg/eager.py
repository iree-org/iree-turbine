# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom op integration into the eager executor."""

from timeit import default_timer
from typing import Optional

import torch

from iree.runtime import (
    HalBufferView,
    HalElementType,
    HalFence,
    VmRef,
    VmVariantList,
)

from ...support.exceptions import (
    UnsupportedTypeError,
)

from ...support.logging import (
    runtime_logger as logger,
)

from ..device import (
    Device,
    lookup_device_from_torch,
)

from ..tracing import tracer

from .base import (
    AttrArg,
    IntArg,
    KernelSelection,
)

from .compiler import (
    compile_standalone_kernel,
    KernelCompileConfig,
)

from ..invoke import (
    invoke_vm_function,
)

__all__ = [
    "eager_dispatch",
]


def eager_dispatch(ksel: KernelSelection):
    """Main entry-point for handling dispatch of a selected kernel via a generator."""
    # Scan arg descs and decide on a compute device.
    # For now, we compute on the first device that we support.
    # This is very simplisitic and will need to be extended for multi-device, etc.
    device: Optional[Device] = None
    torch_device: Optional[torch.device] = None
    for arg_desc in ksel.arg_descs:
        assert arg_desc is not None, "NYI: None arguments"
        if not arg_desc.is_list:
            if arg_desc.ir_arity == 1:
                # One arg has maybe_tensor_value as a single element (common case).
                tensor_arg = arg_desc.maybe_tensor_value
                if tensor_arg is None:
                    continue
                assert isinstance(tensor_arg, torch.Tensor)
                torch_device = tensor_arg.device
                device = lookup_device_from_torch(torch_device)
                if device is not None:
                    break
            else:
                # Optional arg omitted.
                assert arg_desc.ir_arity == 0
                continue
        else:
            # List. maybe_tensor_value is a list. Uncommon case.
            assert isinstance(arg_desc.maybe_tensor_value, list)
            for tensor_arg in arg_desc.maybe_tensor_value:
                if tensor_arg is None:
                    continue
                torch_device = tensor_arg.device
                device = lookup_device_from_torch(torch_device)
                if device is not None:
                    break

    # Default to CPU.
    if device is None:
        logger.debug("Fallback to CPU device due to no supported device in arguments")
        torch_device = torch.device("cpu")
        device = lookup_device_from_torch(torch_device)
        assert (
            device is not None
        ), "Could not resolve lookup_device_from_torch for argument"

    # Compile.
    # TODO: We can do compilation asynchronously with the device movement
    vm_context, vm_f, config = compile_standalone_kernel(device, ksel)

    # Build the concrete args, issuing device movement as necessary.
    arg_list = VmVariantList(len(ksel.arg_descs))

    def push_scalar(scalar_value):
        if isinstance(scalar_value, int):
            arg_list.push_int(scalar_value)
        elif isinstance(scalar_value, float):
            arg_list.push_float(scalar_value)
        else:
            raise UnsupportedTypeError(type(scalar_value))

    def push_tensor(tensor_arg):
        if tensor_arg.device != torch_device:
            # TODO: If the source and target device are both known to us,
            # we can do this "in house" vs asking torch to do it.
            tensor_arg = tensor_arg.to(torch_device)
        if not tensor_arg.is_contiguous():
            if config.layout_specialized:
                raise NotImplementedError(
                    "Layout specialized kernels are not yet implemented"
                )
            tensor_arg = tensor_arg.contiguous()
        # Since we know we are on the same device, we can use the unsafe
        # import_torch_tensor.
        arg_list.push_ref(device.import_torch_tensor(tensor_arg))

    for arg_desc in ksel.arg_descs:
        assert arg_desc is not None, "NYI: None arguments"
        arity = arg_desc.ir_arity
        if not arg_desc.is_list:
            # Non-list.
            if arity == 1:
                tensor_arg = arg_desc.maybe_tensor_value
                if tensor_arg is not None:
                    push_tensor(tensor_arg)
                else:
                    assert isinstance(arg_desc, (IntArg, AttrArg))
                    push_scalar(arg_desc.v)
            else:
                continue
        else:
            # List. Uncommon case.
            tensor_arg = arg_desc.maybe_tensor_value
            if tensor_arg is not None:
                for i in range(arity):
                    push_tensor(tensor_arg[i])
            else:
                for i in range(arity):
                    assert isinstance(arg_desc, (IntArg, AttrArg))
                    list_arg = arg_desc.v
                    assert isinstance(list_arg, list)
                    push_scalar(list_arg[i])

    ret_list = VmVariantList(len(ksel.result_descs))

    invoke_time = invoke_vm_function(
        device,
        config.async_invocations,
        vm_context,
        vm_f,
        arg_list,
        ret_list,
        timer=default_timer,
    )

    if tracer.enabled:
        _log_eager_dispatch(config, arg_list, invoke_time * 1000)

    # Unpack results.
    results = []
    for i, result_desc in enumerate(ksel.result_descs):
        arity = result_desc.ir_arity
        meta_tensor_value = result_desc.maybe_tensor_value
        if meta_tensor_value is None:
            # Scalar return.
            raise NotImplementedError("CustomOp scalar return")
        assert isinstance(
            meta_tensor_value, torch.Tensor
        ), "NYI: Optional and result lists"

        # Tensor return. The meta tensor value already has the correct torch
        # dtype and shape, so we just need to export and return it for the
        # appropriate device.
        bv: HalBufferView = HalBufferView.__iree_vm_cast__(ret_list.get_as_ref(i))
        results.append(device.export_torch_tensor(bv, meta_tensor_value))

    if len(results) == 1:
        return results[0]
    elif len(results) == 0:
        return None
    else:
        return tuple(results)


def _log_eager_dispatch(
    config: KernelCompileConfig, arg_list: VmVariantList, invoke_time_millis: float
):
    args = []
    try:
        for i in range(arg_list.size):
            variant = arg_list.get_variant(i)
            if isinstance(variant, VmRef):
                if variant.isinstance(HalBufferView):
                    args.append(_log_format_buffer_view(variant.deref(HalBufferView)))
                    continue
            args.append(variant)
    except:
        tracer.exception("Exception while pretty-printing arguments")

    msg = ""
    tracer.log_structured(
        tag="INVOKE_KERNEL",
        msg=msg,
        columns=[config.tracing_key, invoke_time_millis] + args,
    )


def _log_format_buffer_view(bv: HalBufferView) -> str:
    # TODO: We should expose this as a method on HalBufferView upstream instead
    # of half doing it here.
    shape = "x".join(str(i) for i in bv.shape)
    dtype_desc = _LOG_HAL_ELEMENT_TYPE_DESC.get(bv.element_type)
    if dtype_desc is None:
        dtype_desc = f"<{bv.element_type}>"
    return f"{shape}x{dtype_desc}"


_LOG_HAL_ELEMENT_TYPE_DESC = {
    HalElementType.BFLOAT_16: "bf16",
    HalElementType.BOOL_8: "i1",
    HalElementType.COMPLEX_64: "cf64",
    HalElementType.COMPLEX_128: "cf128",
    HalElementType.FLOAT_16: "f16",
    HalElementType.FLOAT_32: "f32",
    HalElementType.FLOAT_64: "f64",
    HalElementType.INT_4: "i4",
    HalElementType.INT_8: "i8",
    HalElementType.INT_16: "i16",
    HalElementType.INT_32: "i32",
    HalElementType.INT_64: "i64",
    HalElementType.SINT_4: "si4",
    HalElementType.SINT_8: "si8",
    HalElementType.SINT_16: "si16",
    HalElementType.SINT_32: "si32",
    HalElementType.SINT_64: "si64",
    HalElementType.UINT_4: "ui4",
    HalElementType.UINT_8: "ui8",
    HalElementType.UINT_16: "ui16",
    HalElementType.UINT_32: "ui32",
    HalElementType.UINT_64: "ui64",
}
