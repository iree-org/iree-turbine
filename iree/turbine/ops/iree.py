# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom ops for built-in IREE functionality."""
from contextlib import AbstractContextManager
from copy import deepcopy
from typing import cast, TYPE_CHECKING
import numpy as np
import os
import torch.fx


from ..support.ir_imports import (
    Attribute,
    RankedTensorType,
    StringAttr,
    Value,
    flow_d,
    tensor_d,
)
from ..support.torch import torch_device_equal

from ..runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
    AttrArg,
    def_library,
)

if TYPE_CHECKING:
    from ..aot import DeviceAffinity

from ..support import debugging

__all__ = [
    "trace",
]

IREE_LIBRARY = def_library("iree")

################################################################################
# trace_tensor / trace_tensors
# See the flow.tensor_trace op for details. In essence:
#   * trace_key is a name to label tensors with (intended for log filtering)
#   * tensor or tensors are values to log a value for
################################################################################


@CustomOp.register(library=IREE_LIBRARY)
class trace_tensor(CustomOp):
    signature = "trace_tensor(str trace_key, Tensor tensor) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ksel.arg_tensor(1)

    def eager_execute(self, key, tensor):
        debugging.trace_tensor_callback(key, tensor)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        key = cast(AttrArg, ksel.arg_descs[0])
        _emit_tensor_trace(kb, cast(str, key.v), [kb.arg_bindings[1]])
        kb.yield_results()


# This relies on a nonpublic functionality, so it is likely that it will break in
# future torch versions.
# We really want to make the op side-effecting because this is really the correct
# semantics.
# If we don't have this we need to declare the op as mutating the tensor argument, which
# causes problems when trying to trace frozen tensors.
# It will also cause problems when for example we trace a tensor that does not affect
# the result of the export. Then the trace op will be removed as dead code.
# E.g.
# ```
# y = x.clone()
# trace_tensor(y)
# ```
torch.fx.node._side_effectful_functions.add(
    trace_tensor.default  # type: ignore[attr-defined]
)

################################################################################
# transfer_to_logical_device
# This is a graph-mode meta-data op which transfers a tensor to a local, logical
# device managed by the runtime. In this context, there may or may not be any
# correspondence between such local devices and Torch devices: they may be
# shards of a CPU/GPU, otherwise unaddressable NPU device, abacus, etc.
# The actual device assignment is late bound at compilation time, and the
# reference here is just a logical name.
#
# All Turbine programs have a logical device "0" that is used for anything not
# otherwise annotated. This means that for common use cases like sharding across
# homogenous devices, there can easily be a "1", "2", etc. However, note that
# there is nothing at this level that requires devices to be homogenous or
# named in such a way. Internal to the module, this will require that a symbol
# with the name "__device.{moniker}" is provided in some fashion (spec file,
# command line flags, etc).
#
# Within a graph, transferring tensors to a device causes partitioning and
# optimization of placement of consuming compute operations. Under the covers,
# this maps to:
#   flow.tensor.transfer %t to #hal.device.promise<@moniker>
################################################################################


@CustomOp.register(library=IREE_LIBRARY)
class transfer_to_logical_device(CustomOp):
    signature = "transfer_to_logical_device(str moniker, Tensor tensor) -> Tensor"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ta = ksel.arg_tensor(1)
        spec = [i for i, s in enumerate(ta.t.shape) if isinstance(s, int)]

        ta.specialize_dims(*spec)
        ksel.return_tensor(ta.t).specialize_dims(*spec)

    def eager_execute(self, device_moniker: str, tensor: torch.Tensor):
        if iree_device_moniker_to_torch_device_map is None:
            # Clone to match the semantics in eager.
            # When transferring the result is a new tensor.
            return tensor.clone()

        target_torch_device = iree_device_moniker_to_torch_device_map[device_moniker]
        if torch_device_equal(tensor.device, target_torch_device):
            return tensor.clone()
        else:
            return tensor.to(device=target_torch_device)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        moniker = cast(AttrArg, ksel.arg_descs[0]).v
        t = kb.arg_bindings[1]
        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)
        target = Attribute.parse(f'#hal.device.promise<@"__device_{moniker}">')
        result = flow_d.TensorTransferOp(t, dynamic_dims, target).result
        kb.yield_results(result)


@CustomOp.register(library=IREE_LIBRARY)
class barrier_on_logical_device(CustomOp):
    signature = "barrier_on_logical_device(str moniker, Tensor tensor) -> Tensor"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ta = ksel.arg_tensor(1)
        spec = [i for i, s in enumerate(ta.t.shape) if isinstance(s, int)]

        ta.specialize_dims(*spec)
        ksel.return_tensor(ta.t).specialize_dims(*spec)

    def eager_execute(self, device_moniker, tensor):
        return tensor

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        moniker = cast(AttrArg, ksel.arg_descs[0]).v
        t = kb.arg_bindings[1]
        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)
        target = Attribute.parse(f'#hal.device.promise<@"__device_{moniker}">')
        result = flow_d.TensorBarrierOp(t, dynamic_dims, target).result
        kb.yield_results(result)


################################################################################
# IREE device affinity to torch device map
################################################################################

iree_device_moniker_to_torch_device_map: dict[str, torch.device] | None = None


def set_iree_device_affinity_to_torch_device_map(
    map: dict["DeviceAffinity", torch.device] | None = None,
):
    moniker_map = None
    if map is not None:
        moniker_map = {str(k.ordinal): v for k, v in map.items()}
    global iree_device_moniker_to_torch_device_map
    iree_device_moniker_to_torch_device_map = moniker_map


class IreeDeviceAffinityToTorchDevice(AbstractContextManager):
    """Allows for tensor transfers between devices in eager mode.

    Example:

    .. code-block:: Python

        t = torch.tensor([1, 2], device="cuda:2")
        with IreeDeviceAffinityToTorchDevice({
            DeviceAffinity(0): torch.device("cuda:2"),
            DeviceAffinity(1): torch.device("cuda:3")
        }):
            t2 = transfer_to_logical_device("1", t) # move to cuda:3
            t3 = transfer_to_logical_device("0", t2) # move back to cuda:2
    """

    def __init__(self, map: dict["DeviceAffinity", torch.device] | None):
        self.map = deepcopy(map)
        self._old_map: list[dict[str, torch.device] | None] = []

    def __enter__(self):
        global iree_device_moniker_to_torch_device_map
        self._old_map.append(iree_device_moniker_to_torch_device_map)
        set_iree_device_affinity_to_torch_device_map(self.map)

    def __exit__(self, *excinfo):
        global iree_device_moniker_to_torch_device_map
        iree_device_moniker_to_torch_device_map = self._old_map.pop()


################################################################################
# Emission utilities
################################################################################


def _append_dynamic_dims(kb: KernelBuilder, dynamic_dims: list[Value], tensor: Value):
    rtt = RankedTensorType(tensor.type)
    for i in range(rtt.rank):
        if rtt.is_dynamic_dim(i):
            dynamic_dims.append(tensor_d.dim(tensor, kb.constant_index(i)))


def _emit_tensor_trace(kb: KernelBuilder, key: str, ts: list[Value]):
    dynamic_dims: list[Value] = []
    for t in ts:
        _append_dynamic_dims(kb, dynamic_dims, t)
    flow_d.TensorTraceOp(StringAttr.get(key), ts, dynamic_dims)
