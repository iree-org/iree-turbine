# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha1
from typing import Any, Callable, Tuple

import torch
from torch.fx.graph_module import GraphModule

# from torch._functorch.aot_autograd import
from torch._functorch.partitioners import default_partition
from torch.autograd import Function

from iree.compiler.extras.fx_importer import FxImporter

from .library import register_impl, register_meta, define_schema
from .utils import is_boo_backward_enabled
from ..runtime import get_launchable
from ....runtime.launch import Launchable
from ....dynamo.passes import turbine_cpu_pass_pipeline
from ....transforms.general.custom_op_expansion import ExpandCustomOpsPass

__all__ = [
    "mlir_from_gm",
    "get_io_from_gm",
    "get_schema",
    "get_custom_graph_op",
    "make_autograd_function",
]


def get_io_from_gm(gm):
    """Returns input nodes and output TensorMetadata from the graph module."""
    inputs = []
    meta_outputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.target)
        if node.op == "output":
            meta_outputs.extend(
                [val.meta.get("tensor_meta", None) for val in node.all_input_nodes]
            )
    return inputs, meta_outputs


def get_schema(inputs, outputs):
    """Generate a schema from the result of `get_io_from_gm."""
    schema = "("
    schema += ", ".join([f"Tensor {inp}" for inp in inputs])
    schema += ") -> ("
    schema += ", ".join(["Tensor" for out in outputs])
    schema += ")"
    return schema


def mlir_from_gm(gm: GraphModule) -> Any:
    sample_args = tuple(
        [n.meta.get("val") for n in gm.graph.nodes if n.op == "placeholder"]
    )
    gm = turbine_cpu_pass_pipeline(gm, sample_args)
    imp = FxImporter()
    imp.import_graph_module(gm)
    exp_ops = ExpandCustomOpsPass(imp.module_op)
    exp_ops.run()
    return imp.module_op


def get_custom_graph_op(gm: GraphModule) -> Callable[[Any], Any]:
    """Converts a graph module into a custom operator."""
    inputs, outputs = get_io_from_gm(gm)

    hash = sha1(str(gm).encode(), usedforsecurity=False).hexdigest()

    schema = get_schema(inputs, outputs)
    op_name = f"fused_op_{hash}"

    define_schema(op_name, schema)

    # module_op = mlir_from_gm(gm)
    arg_factory = lambda: tuple(
        [n.meta.get("val") for n in gm.graph.nodes if n.op == "placeholder"]
    )

    @register_impl(op_name)
    def _(*args):
        l = get_launchable(gm, arg_factory=arg_factory, func_name=op_name)
        return l(*[arg.data for arg in args])

    @register_meta(op_name)
    def _meta(*args):
        if len(outputs) == 1:
            return torch.empty_strided(
                list(outputs[0].shape),
                stride=outputs[0].stride,
                dtype=outputs[0].dtype,
                device="meta",
                requires_grad=outputs[0].requires_grad,
            )
        return tuple(
            (
                torch.empty_strided(
                    list(o.shape),
                    stride=o.stride,
                    dtype=o.dtype,
                    device="meta",
                    requires_grad=o.requires_grad,
                )
                for o in outputs
            )
        )

    def _f(*args):
        return getattr(torch.ops.boo, op_name)(*args)

    return _f


def make_autograd_function(
    joint_gm: torch.fx.GraphModule, sample_args, num_fwd_outputs
):
    """From a joint forward/backward graph module, creates an autograd function for calling iree custom graph ops for forward and backward."""
    fwd_g, bwd_g = default_partition(
        joint_module=joint_gm,
        _joint_inputs=sample_args,
        num_fwd_outputs=num_fwd_outputs,
    )
    fwd_launch = get_custom_graph_op(fwd_g)
    bwd_launch = (
        get_custom_graph_op(bwd_g) if is_boo_backward_enabled() else bwd_g.forward
    )

    class _MyOpSample(Function):
        @staticmethod
        def forward(ctx, *args):
            all_outputs = fwd_launch(*[arg for arg in args])
            fwd_outputs = all_outputs[:num_fwd_outputs]
            stash_outputs = all_outputs[num_fwd_outputs:]
            ctx.save_for_backward(*stash_outputs)
            return fwd_outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            stashed_tensors = ctx.saved_tensors
            all_outputs = bwd_launch(*stashed_tensors, *grad_outputs)
            return all_outputs

    return _MyOpSample.apply
