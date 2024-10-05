# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom ops for built-in IREE functionality."""
from typing import cast

from ..support.ir_imports import (
    Attribute,
    RankedTensorType,
    StringAttr,
    Value,
    flow_d,
    tensor_d,
)

from ..runtime.op_reg import (
    CustomOp,
    KernelBuilder,
    KernelSelection,
    AttrArg,
    def_library,
)

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
    signature = "trace_tensor(str trace_key, Tensor(a!) tensor) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ksel.arg_tensor(1, inplace_tied=True)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        key = cast(AttrArg, ksel.arg_descs[0])
        _emit_tensor_trace(kb, cast(str, key.v), [kb.arg_bindings[1]])
        kb.yield_results(kb.arg_bindings[1])


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
# Within a graph, transfering tensors to a device causes partitioning and
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
        ksel.return_tensor(ta.t)

    def eager_execute(self, device_moniker, tensor):
        return tensor

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        moniker = cast(AttrArg, ksel.arg_descs[0]).v
        t = kb.arg_bindings[1]
        dynamic_dims: list[Value] = []
        _append_dynamic_dims(kb, dynamic_dims, t)
        target = Attribute.parse(f'#hal.device.promise<@"__device_{moniker}">')
        result = flow_d.TensorTransferOp(t, dynamic_dims, target).result
        kb.yield_results(result)


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
