# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hashlib import sha1
from typing import Sequence, Literal

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.node import Target

from .library import *
from .utils import (
    MemoryFormatPermutation,
    get_arg_spec_name_and_memory_format_permutations,
    get_memory_format_permutation,
)
from ..runtime import get_launchable
from ....support.logging import aot_logger as logger

__all__ = [
    "get_custom_graph_op",
]


def _get_io_from_gm(
    gm: GraphModule,
) -> tuple[list[Target], list[torch.Tensor | None]]:
    """Returns input nodes and output fake tensors from the graph module."""

    inputs = []
    meta_outputs = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            inputs.append(node.target)
        if node.op == "output":
            meta_outputs.extend(
                [
                    val.meta.get("val", None) if val is not None else None
                    for val in node.args[0]
                ]
            )
    return inputs, meta_outputs


def _get_schema(
    inputs: Sequence[Target], outputs: Sequence[torch.Tensor | None]
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
    """Converts a graph module into a custom operator.

    This function infers input/output signature from the graph metadata, and produces a specialized op.
    The returned op will not automatically re-specialize for different inputs.
    """
    gm_string = str(gm.print_readable(print_output=False, include_stride=True))
    hash = sha1(gm_string.encode(), usedforsecurity=False).hexdigest()
    call_function_names = "_".join(
        n.name for n in gm.graph.nodes if n.op == "call_function"
    )

    # Evidently, there is a limit to the number of characters in a path.
    # We use this name for the file cache, so some modest limits need to be set.
    # TODO: reorganize the file cache so this isn't problematic.
    op_name = (
        f"fused_op_{call_function_names}_{hash}"
        if len(call_function_names) < 120
        else f"fused_op_{hash}"
    )
    logger.debug("Got hash str '%s' for GraphModule: \n %s", hash, gm_string)

    if not hasattr(torch.ops.boo, op_name):
        _define_custom_graph_op(
            gm, op_name, force_single_dispatch=force_single_dispatch
        )

    return get_library_op(op_name)


def _handle_layouts(
    args: Sequence[torch.Tensor],
    perms: Sequence[MemoryFormatPermutation | None],
    perm_item: Literal["permutation", "inverse_permutation"],
) -> tuple[torch.Tensor, ...]:
    """Applies torch.permute(arg[i], perms[i].perm_item) to all args."""
    return tuple(
        [
            arg if perm is None else arg.permute(getattr(perm, perm_item))
            for perm, arg in zip(perms, args, strict=True)
        ]
    )


class _LayoutManagedModuleForAOTMlirExport(torch.nn.Module):
    """This conjugates the forward call of a source module with permutations.

    The use of this module is the following:

    1. In an aot situation, identify non-contiguous inputs and outputs.
    2. For each non-contiguous input/output, identify a shape permutation which would result in a contiguous tensor.
    3. Outside the function boundary given to IREE, permute the input tensors to contiguous format.
    4. Inside this module (which is to be compiled with IREE), the inverse permutations are applied to inputs.
    5. To the outputs, we also apply the forward permutations inside this module.
    6. Since the outputs of IREE are always contiguous, those permutations produce outputs with data stored in the desired layout.
    7. In pytorch, permute the outputs of IREE back to the original shape.
    """

    def __init__(
        self,
        input_mem_format_perms: Sequence[MemoryFormatPermutation | None],
        output_mem_format_perms: Sequence[MemoryFormatPermutation | None],
        source_module: torch.nn.Module,
    ):
        super().__init__()
        self.input_mem_format_perms = input_mem_format_perms
        self.output_mem_format_perms = output_mem_format_perms
        self.src_module = source_module

    def forward(self, *args):
        handled_args = _handle_layouts(
            args, perms=self.input_mem_format_perms, perm_item="inverse_permutation"
        )
        outputs = self.src_module(*handled_args)
        single_output = False
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            single_output = True
        handled_outputs = _handle_layouts(
            outputs, perms=self.output_mem_format_perms, perm_item="permutation"
        )
        return handled_outputs[0] if single_output else tuple(handled_outputs)


def _define_custom_graph_op(
    gm: GraphModule, op_name: str, *, force_single_dispatch: bool = False
):
    """Defines a custom op from the graph module with given op_name in the boo library."""
    inputs, outputs = _get_io_from_gm(gm)
    is_none_output = _maybe_trim_none_outputs(gm)
    has_a_none_output = any(is_none_output)
    schema = _get_schema(inputs, outputs)
    define_schema(op_name, schema)
    # Get memory format permutations for output tensors based on graph metadata.
    output_mem_format_perms = [
        get_memory_format_permutation(t, strict=True) for t in outputs if t is not None
    ]
    logger.debug(
        "Output fake tensors:\n%s\nOutput MemoryFormatPermutation:\n%s",
        str(outputs),
        str(output_mem_format_perms),
    )

    input_fake_tensors: list[torch.Tensor | None] = [
        n.meta.get("val") for n in gm.graph.find_nodes(op="placeholder")
    ]

    assert all(
        [t is not None for t in input_fake_tensors]
    ), f"Expected fake input tensors for graph module:\n{gm}\nGot {input_fake_tensors}."

    spec_name, input_mem_format_perms = (
        get_arg_spec_name_and_memory_format_permutations(op_name, *input_fake_tensors)
    )

    @register_impl(op_name)
    def _(*args):
        handled_inputs = _handle_layouts(
            args, perms=input_mem_format_perms, perm_item="permutation"
        )
        launch = get_launchable(
            lambda: _LayoutManagedModuleForAOTMlirExport(
                input_mem_format_perms=input_mem_format_perms,
                output_mem_format_perms=output_mem_format_perms,
                source_module=gm,
            ),
            lambda: handled_inputs,
            func_name=spec_name,
            force_single_dispatch=force_single_dispatch,
        )
        outputs = launch(*[arg.data for arg in handled_inputs])
        single_output = False
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
            single_output = True
        handled_outputs = _handle_layouts(
            outputs, perms=output_mem_format_perms, perm_item="inverse_permutation"
        )
        if not has_a_none_output:
            return handled_outputs[0] if single_output else handled_outputs
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
