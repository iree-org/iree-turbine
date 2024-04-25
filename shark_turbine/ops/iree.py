# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom ops for built-in IREE functionality."""
from typing import cast

from ..support.ir_imports import (
    Operation,
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
    signature = "trace_tensor(str trace_key, Tensor(a!) tensor) -> ()"

    def select(self, ksel: KernelSelection):
        ksel.attr_str(0)
        ksel.arg_tensor(1, inplace_tied=True)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        key = cast(AttrArg, ksel.arg_descs[0])
        _emit_tensor_trace(kb, cast(str, key.v), [kb.arg_bindings[1]])
        kb.yield_results(kb.arg_bindings[1])


@CustomOp.register(library=IREE_LIBRARY)
class _test_add(CustomOp):
    signature = "_test_add(Tensor t1, Tensor t2) -> (Tensor)"

    def select(self, ksel: KernelSelection):
        t1_desc = ksel.arg_tensor(0)
        t1_desc.specialize_all_dims()
        t2_desc = ksel.arg_tensor(1)
        t2_desc.specialize_all_dims()
        result_desc = ksel.return_new_tensor(list(t1_desc.t.shape), t1_desc.t.dtype)
        result_desc.specialize_all_dims()

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        t1, t2 = kb.arg_bindings
        result_type = t1.type  # type: ignore
        result = Operation.create(
            "tosa.add", results=[result_type], operands=[t1, t2]
        ).result
        kb.yield_results(result)
