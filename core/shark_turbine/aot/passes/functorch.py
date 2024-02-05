# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

import torch
from torch.fx import (
    GraphModule,
)
from torch.fx.experimental import proxy_tensor
from torch.utils import _pytree as pytree


# Use the functorch `functionalize()` helper. That cannot be used directly
# because it does not correctly handle fake tensor tracing. But we use
# the underlying dispatcher controls to enable/disable it and perform
# the transform. The approach was lifted from what ONNX is doing and a
# number of issues. In its present form it has a number of issues:
#   1. Cannot trace structured inputs and will drop output signature
#      rewrites usually done by torch.export, rending structured
#      results a non-starter if used as a transform after torch.export.
#   2. Will not play nicely with an enclosing, user specified fake mode.
#      There is a lot of code on the ONNX side to enable this, but I
#      don't have test cases for it and don't want to just blindly
#      adapt dead code.
#   3. Loses backtrace information. The ONNX side has a helper that
#      re-associates this, but it wasn't obvious it would work in our
#      exact scenario.
# Further, it is not clear at all why this is using such heavy-weight
# facilities to do a simple graph transformation. I expect that we just
# need to write a pure FX pass to do the functionalization transform to
# our liking and shoot this into the sun. If we spend any time at all
# debugging the issues that can arise from all of this layering, we
# should just do that.
#
# For the reasons above, we only use this as a *pre-export* transformation,
# since that does not result in load bearing information loss. Note that
# ONNX applies this post export, which suffers from the loss of output
# destructuring rewrites that torch.export does.
def functorch_functionalize(gm: GraphModule, *args) -> GraphModule:
    functionalized_callable = _functionalize_callabale(gm)
    # TODO: There is more of a dance needed if the user has entered with a fake_mode.
    with proxy_tensor.maybe_disable_fake_tensor_mode():
        new_gm = proxy_tensor.make_fx(
            functionalized_callable,
            decomposition_table={},
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _allow_fake_constant=False,
        )(*args)

    return new_gm


def _functionalize_callabale(function: Callable) -> Callable:
    def wrapped(*args):
        args_functional = pytree.tree_map_only(
            torch.Tensor, torch._to_functional_tensor, args
        )
        torch._enable_functionalization(reapply_views=True)
        try:
            out = function(*args_functional)
        finally:
            torch._disable_functionalization()
        # Do a dance to re-associate inputs.
        flat_inputs, _ = pytree.tree_flatten(args)
        flat_inputs_functional, _ = pytree.tree_flatten(args_functional)
        for input_raw, input_functional in zip(flat_inputs, flat_inputs_functional):
            if isinstance(input_functional, torch.Tensor):
                torch._sync(input_functional)
                torch._from_functional_tensor(input_functional)
        pytree.tree_map_only(torch.Tensor, torch._sync, out)
        out_unwrapped = pytree.tree_map(torch._from_functional_tensor, out)
        return out_unwrapped

    return wrapped
