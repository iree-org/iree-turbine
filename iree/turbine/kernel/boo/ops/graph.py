# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha1
from typing import Any, Callable, Tuple, Sequence

import torch
from torch.fx.graph_module import GraphModule

from torch._functorch.aot_autograd import aot_export_joint_simple
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
    # "mlir_from_gm",
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
            print(f"{node.args=}")
            meta_outputs.extend(
                [
                    val.meta.get("tensor_meta", None) if val is not None else None
                    for val in node.args[0]
                ]
            )
    return inputs, meta_outputs


def get_schema(inputs, outputs):
    """Generate a schema from the result of `get_io_from_gm."""

    ret_ty = "Tensor?" if any([o is None for o in outputs]) else "Tensor"
    schema = "("
    schema += ", ".join([f"Tensor {inp}" for inp in inputs])
    schema += ") -> ("
    schema += ", ".join([ret_ty for out in outputs])
    schema += ")"
    return schema


# def mlir_from_gm(gm: GraphModule) -> Any:
#     sample_args = tuple(
#         [n.meta.get("val") for n in gm.graph.nodes if n.op == "placeholder"]
#     )
#     gm = turbine_cpu_pass_pipeline(gm, sample_args)
#     imp = FxImporter()
#     imp.import_graph_module(gm)
#     exp_ops = ExpandCustomOpsPass(imp.module_op)
#     exp_ops.run()
#     return imp.module_op


def tensor_type_str(t: torch.Tensor | None) -> str:
    if t is None:
        return ""
    shape = t.shape
    dtype = str(t.dtype).removeprefix("torch.")
    shape_str = "x".join([str(dim) for dim in shape])
    return shape_str + f"x{dtype}"


def get_name(base_name, *args):
    name = base_name
    for idx, arg in enumerate(args):
        if arg is not None and not isinstance(arg, torch.Tensor):
            raise TypeError(
                f"Expected all function arguments to be (optional) tensors. Got {type(arg)} at position {idx}."
            )
        name += f"_{tensor_type_str(arg)}"
    return name


def get_custom_graph_op(gm: GraphModule) -> Callable[[Any], Any]:
    """Converts a graph module into a custom operator."""
    inputs, outputs = get_io_from_gm(gm)
    # TODO: handle this better
    is_none_output = maybe_trim_none_outputs(gm)
    has_a_none_output = any(is_none_output)

    hash = sha1(str(gm).encode(), usedforsecurity=False).hexdigest()

    schema = get_schema(inputs, outputs)
    op_name = f"fused_op_{hash}"
    print(f"{op_name}::{schema}")

    define_schema(op_name, schema)

    @register_impl(op_name)
    def _(*args):
        spec_name = get_name(op_name, *args)
        l = get_launchable(gm, arg_factory=args, func_name=spec_name)
        results = l(*[arg.data for arg in args])
        if not has_a_none_output:
            return results
        # Handle this better.
        if isinstance(results, torch.Tensor):
            results = [results]
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

    custom_library_op = getattr(torch.ops.boo, op_name)

    def _f(*args):
        return custom_library_op(*args)

    return _f


def maybe_trim_none_outputs(gm: GraphModule) -> Sequence[bool]:
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


def make_autograd_function(
    joint_gm: torch.fx.GraphModule, sample_args, num_fwd_outputs
):
    """From a joint forward/backward graph module, creates an autograd function for calling iree custom graph ops for forward and backward."""
    print("Partitioning subgraph")
    fwd_g, bwd_g = default_partition(
        joint_module=joint_gm,
        _joint_inputs=sample_args,
        num_fwd_outputs=num_fwd_outputs,
    )
    fwd_g.print_readable()
    bwd_g.print_readable()
    bwd_g.print_readable()

    fwd_launch = get_custom_graph_op(fwd_g)
    bwd_launch = (
        get_custom_graph_op(bwd_g) if is_boo_backward_enabled() else bwd_g.forward
    )

    print("defining autograd function")

    class _MyOpSample(Function):
        @staticmethod
        def forward(ctx, *args):
            all_outputs = fwd_launch(*args)
            fwd_outputs = all_outputs[:num_fwd_outputs]
            stash_outputs = all_outputs[num_fwd_outputs:]
            ctx.save_for_backward(*stash_outputs)
            return fwd_outputs

        @staticmethod
        def backward(ctx, *grad_outputs):
            stashed_tensors = ctx.saved_tensors
            all_outputs = bwd_launch(*stashed_tensors, *grad_outputs)
            return all_outputs

    def _f(*args):
        return _MyOpSample.apply(*args)

    return _f
