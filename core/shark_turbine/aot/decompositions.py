# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
from typing import Optional

import torch

from ..dynamo.decompositions import (
    _current,
    _extend_context_manager,
    DecompositionOpsList,
    DecompositionTable,
)

__all__ = [
    "current_aot_decompositions",
    "extend_aot_decompositions",
]


def current_aot_decompositions() -> DecompositionTable:
    """Gets the current decomposition table for AOT."""
    return _current("aot")


def extend_aot_decompositions(
    *,
    from_current: bool = True,
    add_ops: Optional[DecompositionOpsList] = None,
    remove_ops: Optional[DecompositionOpsList] = None
):
    """Context manager which extends the list of decompositions used for AOT."""
    return _extend_context_manager(
        "aot", from_current=from_current, add_ops=add_ops, remove_ops=remove_ops
    )


###############################################################################
# Workarounds
###############################################################################


def _patch_op_dispatch(op):
    if torch.__version__ >= "2.3.0" and torch.__version__ < "2.4":
        # Around the torch 2.3.0 release cut, there was a regression such that
        # running decompositions in a functionalized context did not work
        # with Python registered ops. The issue is that they have an incomplete
        # list of mode handler registrations and cannot handle the
        # FunctionalTensorMode. Since we only have a handful of these, and
        # since we can assume that for the sake of expediency, functional
        # dispatch is basically the same as fake tensor dispatch, we just
        # take the fake tensor registration and dup it onto the functional
        # registration.
        # Note that the torch._higher_order_ops.auto_functionalize is registered
        # in Python and is itself broken, it needs to be monkey patched.
        # See: https://github.com/pytorch/pytorch/issues/122752
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        t = op.python_key_mode_table
        if FunctionalTensorMode not in t:
            handler = t[FakeTensorMode]
            t[FunctionalTensorMode] = handler


_patched_op_dispatch_for_export = False


def _patch_op_dispatch_for_export():
    global _patched_op_dispatch_for_export
    if _patched_op_dispatch_for_export:
        return
    _patched_op_dispatch_for_export = True
    import torch._higher_order_ops.auto_functionalize

    _patch_op_dispatch(torch._higher_order_ops.auto_functionalize.auto_functionalized)
