# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha1
from typing import Sequence

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target
from torch.fx.passes.shape_prop import TensorMetadata

from .library import *
from .utils import (
    MemoryFormatInformation,
    get_arg_spec_name_and_memory_format_information,
    get_memory_format_information,
)
from ..runtime import get_launchable
from ....support.logging import aot_logger as logger

__all__ = [
    "get_custom_graph_op",
]


def _get_io_from_gm(
    gm: GraphModule,
) -> tuple[list[Target], list[TensorMetadata | None]]:
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


def _get_schema(
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


def get_custom_graph_op(
    gm: GraphModule, *, force_single_dispatch: bool = False
) -> torch._ops.OpOverloadPacket:
    """Converts a graph module into a custom operator."""
    hash = sha1(str(gm).encode(), usedforsecurity=False).hexdigest()
    op_name = f"fused_op_{hash}"
    logger.debug("Got hash str '%s' for GraphModule: \n %s", hash, str(gm))

    if not hasattr(torch.ops.boo, op_name):
        _define_custom_graph_op(
            gm, op_name, force_single_dispatch=force_single_dispatch
        )

    return get_library_op(op_name)


def _define_custom_graph_op(
    gm: GraphModule, op_name: str, *, force_single_dispatch: bool = False
):
    """Defines a custom op from the graph module with given op_name in the boo library."""
    inputs, outputs = _get_io_from_gm(gm)
    is_none_output = _maybe_trim_none_outputs(gm)
    has_a_none_output = any(is_none_output)
    schema = _get_schema(inputs, outputs)
    define_schema(op_name, schema)
    # Get memory format information about the output tensors from the graph metadata.
    output_mem_format_infos = [
        get_memory_format_information(t) for t in outputs if t is not None
    ]
    logger.debug("Output MemoryFormatInformation:\n%s", str(output_mem_format_infos))

    class LayoutManagedModule(torch.nn.Module):
        """This wrapper around gm.forward is the content being offloaded to IREE.

        We are expecting inputs to be permuted into a contiguous format before reaching this point.
        The forward method performs the inverse permutations, which will be imported into MLIR.

        A similar handling of output tensors is performed.
        """

        def __init__(
            self,
            mem_format_infos: Sequence[MemoryFormatInformation],
        ):
            super().__init__()
            self.mem_format_infos = mem_format_infos

        def forward(self, *args):
            handled_args = []
            for idx, arg in enumerate(args):
                arg_mem_format_info = self.mem_format_infos[idx]
                if not arg_mem_format_info.is_channels_last:
                    handled_args.append(arg)
                    continue
                handled_args.append(
                    arg.permute(arg_mem_format_info.inverse_permutation)
                )
            outputs = gm.forward(*handled_args)
            single_output = False
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
                single_output = True
            handled_outputs = []
            for idx, o in enumerate(outputs):
                o_mem_format_info = output_mem_format_infos[idx]
                if not o_mem_format_info.is_channels_last:
                    handled_outputs.append(o)
                    continue
                handled_outputs.append(o.permute(o_mem_format_info.permutation))
            return handled_outputs[0] if single_output else tuple(handled_outputs)

    @register_impl(op_name)
    def _(*args):

        spec_name, mem_format_infos = get_arg_spec_name_and_memory_format_information(
            op_name, *args
        )

        logger.debug("Memory format infos:\n%s", str(mem_format_infos))

        handled_args = []
        for idx, arg in enumerate(args):
            arg_mem_format_info = mem_format_infos[idx]
            if not arg_mem_format_info.is_channels_last:
                handled_args.append(arg)
                continue
            handled_args.append(arg.permute(arg_mem_format_info.permutation))

        l = get_launchable(
            lambda: LayoutManagedModule(mem_format_infos),
            arg_factory=tuple(handled_args),
            func_name=spec_name,
            force_single_dispatch=force_single_dispatch,
        )

        single_output = False
        outputs = l(*[arg.data for arg in handled_args])
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            single_output = True
        handled_outputs = []
        for idx, o in enumerate(outputs):
            o_mem_format_info = output_mem_format_infos[idx]
            if not o_mem_format_info.is_channels_last:
                handled_outputs.append(o)
                continue
            handled_outputs.append(o.permute(o_mem_format_info.inverse_permutation))

        if not has_a_none_output:
            return handled_outputs[0] if single_output else tuple(handled_outputs)
        # We have at least one None output that needs to be included.
        # Handle this better.
        all_results = []
        i = 0
        for is_none in is_none_output:
            if is_none:
                all_results.append(None)
            else:
                all_results.append(handled_outputs[i])
                i += 1
        # It is probably safe to assume all_results has more than one value.
        return tuple(all_results) if i > 1 else all_results[0]

    @register_meta(op_name)
    def _meta(*args):
        outputs = gm.forward(*args)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def _maybe_trim_none_outputs(gm: GraphModule) -> list[bool]:
    """Removes None outputs from graph. The ith return indicates whether output[i] was None."""

    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]

    assert (
        len(output_nodes) == 1
    ), f"Expected single output node for graph module:\n{gm.print_readable(print_output=False)}\nFound {output_nodes = }."

    n = output_nodes[0]
    trunc_returns = [ret for ret in n.args[0] if ret is not None]
    none_output = [ret is None for ret in n.args[0]]

    if not any(none_output):
        return none_output

    new_args = (tuple(trunc_returns),) + n.args[1:]
    n.args = new_args

    gm.graph.lint()
    gm.recompile()
    return none_output
