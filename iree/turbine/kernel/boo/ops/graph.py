# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha1
from typing import Sequence, Callable

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target
from torch.fx.passes.shape_prop import TensorMetadata

from torch._functorch.partitioners import default_partition
from torch.autograd import Function

from iree.compiler.extras.fx_importer import FxImporter

from .library import *
from .utils import is_boo_backward_enabled, get_arg_spec_name
from ..runtime import get_launchable
from ....dynamo.passes import turbine_cpu_pass_pipeline
from ....transforms.general.custom_op_expansion import ExpandCustomOpsPass
from ....support.logging import aot_logger as logger
from ....support.ir_imports import Operation

__all__ = [
    "get_io_from_gm",
    "get_schema",
    "get_custom_graph_op",
    "define_custom_graph_op",
    "get_mlir_module",
    "get_autograd_function",
]


def get_io_from_gm(
    gm: GraphModule,
) -> tuple[Sequence[Target], Sequence[TensorMetadata | None]]:
    """Returns input nodes and output TensorMetadata from the graph module."""

    inputs = []
    meta_outputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.target)
        if node.op == "output":
            meta_outputs.extend(
                [
                    val.meta.get("tensor_meta", None) if val is not None else None
                    for val in node.args[0]
                ]
            )
    return inputs, meta_outputs


def get_schema(
    inputs: Sequence[Target], outputs: Sequence[TensorMetadata | None]
) -> str:
    """Generate a schema from the result of `get_io_from_gm`."""

    ret_ty = "Tensor?" if any([o is None for o in outputs]) else "Tensor"
    schema = "("
    schema += ", ".join([f"Tensor {inp}" for inp in inputs])
    schema += ") -> ("
    schema += ", ".join([ret_ty for _ in outputs])
    schema += ")"
    return schema


def get_mlir_module(gm: GraphModule) -> Operation:
    """Generates torch-mlir IR from a graph module."""
    sample_args = tuple(
        [n.meta.get("val") for n in gm.graph.nodes if n.op == "placeholder"]
    )
    if any([arg is None for arg in sample_args]):
        raise ValueError("Provided gm does not have sufficient metadata.")
    gm = turbine_cpu_pass_pipeline(gm, sample_args)
    imp = FxImporter()
    imp.import_graph_module(gm)
    exp_ops = ExpandCustomOpsPass(imp.module_op)
    exp_ops.run()
    return imp.module_op


def get_custom_graph_op(
    gm: GraphModule, *, force_single_dispatch: bool = False
) -> torch._ops.OpOverloadPacket:
    """Converts a graph module into a custom operator."""
    hash = sha1(str(gm).encode(), usedforsecurity=False).hexdigest()
    op_name = f"fused_op_{hash}"

    if not hasattr(torch.ops.boo, op_name):
        define_custom_graph_op(gm, op_name, force_single_dispatch=force_single_dispatch)

    return get_library_op(op_name)


def define_custom_graph_op(
    gm: GraphModule, op_name: str, *, force_single_dispatch: bool = False
):
    """Defines a custom op from the graph module with given op_name in the boo library."""
    inputs, outputs = get_io_from_gm(gm)
    # TODO: handle this better
    is_none_output = maybe_trim_none_outputs(gm)
    has_a_none_output = any(is_none_output)
    schema = get_schema(inputs, outputs)
    define_schema(op_name, schema)

    @register_impl(op_name)
    def _(*args):
        spec_name = get_arg_spec_name(op_name, *args)
        l = get_launchable(
            gm,
            arg_factory=args,
            func_name=spec_name,
            force_single_dispatch=force_single_dispatch,
        )
        results = l(*[arg.data for arg in args])
        if not has_a_none_output:
            return results
        # We have at least one None output that needs to be included.
        if isinstance(results, torch.Tensor):
            results = [results]
        # Handle this better.
        all_results = []
        i = 0
        for is_none in is_none_output:
            if is_none:
                all_results.append(None)
            else:
                all_results.append(results[i])
                i += 1
        # It is probably safe to assume all_results has more than one value.
        return tuple(all_results) if i > 1 else all_results[0]

    @register_meta(op_name)
    def _meta(*args):
        outputs = gm.forward(*args)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def maybe_trim_none_outputs(gm: GraphModule) -> Sequence[bool]:
    """Removes None outputs from graph. The ith return indicates whether output[i] was None."""
    none_output = []
    for n in gm.graph.nodes:
        if n.op == "output":
            trunc_returns = []
            for ret in n.args[0]:
                if ret is None:
                    none_output.append(True)
                    continue
                trunc_returns.append(ret)
                none_output.append(False)
            new_args = (tuple(trunc_returns),) + n.args[1:]
            n.args = new_args
        gm.graph.lint()
        gm.recompile()
    return none_output


def get_autograd_function(
    joint_gm: torch.fx.GraphModule,
    sample_args: None | tuple[torch.Tensor, ...],
    num_fwd_outputs: int,
    *,
    force_single_dispatch: bool = False,
) -> Callable:
    """From a joint forward/backward graph module, creates an autograd function for calling iree custom graph ops for forward and backward."""
    fwd_g, bwd_g = default_partition(
        joint_module=joint_gm,
        _joint_inputs=sample_args,
        num_fwd_outputs=num_fwd_outputs,
    )
    logger.debug(
        "Partitioned joint graph module into:\nForward:\n%s\nBackward:\n%s",
        str(fwd_g.print_readable(print_output=False)),
        str(bwd_g.print_readable(print_output=False)),
    )

    fwd_launch = get_custom_graph_op(fwd_g, force_single_dispatch=force_single_dispatch)
    # We should handle backward custom graph ops slightly differently.
    # An option could be using the fusion schema to determine which backward ops to hand off to IREE.
    # With the current approach, it doesn't make sense to force the backward into a single dispatch.
    bwd_launch = (
        get_custom_graph_op(bwd_g, force_single_dispatch=False)
        if is_boo_backward_enabled()
        else bwd_g.forward
    )

    class _GeneratedGraphOp(Function):
        @staticmethod
        def forward(ctx, *args):
            all_outputs = fwd_launch(*args)
            if isinstance(all_outputs, torch.Tensor):
                all_outputs = (all_outputs,)
            fwd_outputs = all_outputs[:num_fwd_outputs]
            stash_outputs = all_outputs[num_fwd_outputs:]
            ctx.save_for_backward(*stash_outputs)
            return fwd_outputs[0] if num_fwd_outputs == 1 else fwd_outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            stashed_tensors = ctx.saved_tensors
            all_outputs = bwd_launch(*stashed_tensors, *grad_outputs)
            return all_outputs

    def _f(*args):
        return _GeneratedGraphOp.apply(*args)

    # Hacky function rename to make the replacement graph more descriptive.
    _f.__name__ = f"generated_autograd_{fwd_launch._qualified_op_name}"

    return _f
