# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom ops for built-in IREE functionality."""

from ..support.ir_imports import (
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


def _emit_tensor_trace(kb: KernelBuilder, key: str, ts: list[Value]):
    dynamic_dims = []
    for t in ts:
        rtt = RankedTensorType(t.type)
        for i in range(rtt.rank):
            if rtt.is_dynamic_dim(i):
                dynamic_dims.append(tensor_d.dim(t, kb.constant_index(i)))
    flow_d.TensorTraceOp(StringAttr.get(key), ts, dynamic_dims)


@CustomOp.register(library=IREE_LIBRARY)
class trace_tensor(CustomOp):
    signature = "trace_tensor(str trace_key, Tensor tensor) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ksel.arg_tensor(1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        _emit_tensor_trace(kb, ksel.arg_descs[0].v, [kb.arg_bindings[1]])
        kb.yield_results()


@CustomOp.register(library=IREE_LIBRARY)
class trace_tensors(CustomOp):
    signature = "trace_tensors(str trace_key, Tensor[] tensors) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ksel.arg_tensor_list(1)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        ts = kb.arg_bindings[1]
        if len(ts) >= 1:
            _emit_tensor_trace(kb, ksel.arg_descs[0].v, ts)
        kb.yield_results()
