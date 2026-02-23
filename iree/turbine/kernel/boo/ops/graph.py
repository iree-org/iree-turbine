# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from typing import Sequence, Literal, TypeAlias, Callable, Any

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputSpec,
    OutputSpec,
    InputKind,
    OutputKind,
    TensorArgument,
)
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Target, Node
from torch.fx.passes.shape_prop import TensorMetadata

from .library import *
from .utils import (
    MemoryFormatPermutation,
    get_arg_spec_name_and_memory_format_permutations,
    get_arg_spec_name,
    get_memory_format_permutation,
)
from ..runtime.launch import get_launchable_from_traced_object
from ....support.logging import aot_logger as logger

__all__ = [
    "get_custom_graph_op",
]


class BoundaryClassification(Enum):
    # Standard input
    USER_INPUT = 0
    # An unused input
    UNUSED_INPUT = 1
    # A mutable input
    MUTABLE_INPUT = 2

    # Standard output
    UNIQUE_OUTPUT = 3
    # A repeat of a prior output
    REPEATED_OUTPUT = 4
    # An output of `None`
    NONE_OUTPUT = 5
    # An output which trivially returns an input
    NO_OP_OUTPUT = 6
    # An output which shares the same data as an input
    VIEW_ONLY_OUTPUT = 7

    def is_placeholder(self) -> bool:
        return self.value < 3

    def is_output(self) -> bool:
        return not self.is_placeholder()


@dataclass
class BoundaryValue:
    value: Node | None
    classification: BoundaryClassification
    index: int
    # How to get this boundary value from src_inputs + inner_outputs.
    composition_rule: Callable[[Sequence[Any], Sequence[Any]], Any]


class GraphSchema:
    """
    Wrapper struct around a GraphModule that stores metadata about graph inputs and outputs.

    To automatically determine the metadata from an existing GraphModule, don't initialize directly.
    Instead, use the staticmethod `your_schema = GraphSchema.from_gm(your_gm)` to automatically
    infer the boundary metadata.
    """

    def __init__(
        self,
        placeholders: list[BoundaryValue],
        outputs: list[BoundaryValue],
        src_gm: GraphModule,
    ):
        self.src_gm = src_gm
        self.placeholders = placeholders
        self.outputs = outputs

    @staticmethod
    def from_gm(src_gm: GraphModule) -> "GraphSchema":
        """Creates a GraphSchema from src_gm and infers some input/output metadata."""
        schema = GraphSchema([], [], src_gm)
        schema._initialize()
        return schema

    def get_inner_boundary(self) -> tuple[list[BoundaryValue], list[BoundaryValue]]:
        """Returns a minimal subset of inputs and outputs required for the graph to reconstruct original signature."""
        inner_inputs = list(
            pl
            for pl in self.placeholders
            if pl.classification != BoundaryClassification.UNUSED_INPUT
        )
        inner_outputs = list(
            out
            for out in self.outputs
            if out.classification == BoundaryClassification.UNIQUE_OUTPUT
        )
        return inner_inputs, inner_outputs

    def _initialize(self):
        """Used to populate `self.placeholders` and `self.outputs` from `self.src_gm`."""
        assert len(self.placeholders) == 0, "Must have empty placeholders list."
        assert len(self.outputs) == 0, "Must have empty outputs list."
        placeholder_nodes = list(
            self.src_gm.graph.find_nodes(op="placeholder", sort=True)
        )
        for index, pl in enumerate(placeholder_nodes):
            assert isinstance(pl, Node), f"Expected placeholder, {pl}, to be a Node."
            bv = BoundaryValue(
                pl,
                (
                    BoundaryClassification.UNUSED_INPUT
                    if len(pl.users) == 0
                    else BoundaryClassification.USER_INPUT
                ),
                index,
                (lambda index: (lambda ins, outs: ins[index]))(index),
            )
            self.placeholders.append(bv)
        output_node = self.src_gm.graph.output_node()
        unique_outputs: dict[Node, int] = {}
        for index, o in enumerate(output_node.args[0]):
            maybe_output_index = unique_outputs.get(o, None)
            if maybe_output_index is not None:
                self.outputs.append(
                    BoundaryValue(
                        o,
                        BoundaryClassification.REPEATED_OUTPUT,
                        index,
                        (lambda index: (lambda ins, outs: outs[index]))(
                            maybe_output_index
                        ),
                    )
                )
                continue
            if o is None:
                self.outputs.append(
                    BoundaryValue(
                        o,
                        BoundaryClassification.NONE_OUTPUT,
                        index,
                        lambda ins, outs: None,
                    )
                )
                continue
            if o in placeholder_nodes:
                self.outputs.append(
                    BoundaryValue(
                        o,
                        BoundaryClassification.NO_OP_OUTPUT,
                        index,
                        (lambda index: lambda ins, outs: ins[index])(
                            placeholder_nodes.index(o)
                        ),
                    )
                )
                continue
            # TODO: detect view-only outputs?
            # New non-trivial unique output.
            new_unique_idx = len(unique_outputs.keys())
            self.outputs.append(
                BoundaryValue(
                    o,
                    BoundaryClassification.UNIQUE_OUTPUT,
                    index,
                    (lambda index: lambda ins, outs: outs[index])(new_unique_idx),
                )
            )
            unique_outputs[o] = new_unique_idx


@dataclass
class GraphTransformation:
    """This struct stores the arrows of a boundary transformation:

    Src Inputs ----- src_gm --------> Src Outputs
        |                                 ^
    input_mod                             |
        |                             output_mod**
        V                                 |
    Mod Inputs ----- mod_gm --------> Mod Outputs

    Two transformations can be composed with *.

    **Note: To capture trivial returns being hoisted from graphs, we require
    that `output_mod` takes in `(Src Inputs, Mod Outputs)` and returns `Src Outputs`.
    """

    src_gm: GraphModule
    mod_gm: GraphModule
    input_mod: Callable[[list[torch.Tensor]], list[torch.Tensor]]
    output_mod: Callable[[list[torch.Tensor], list[torch.Tensor]], list[torch.Tensor]]

    def __mul__(self, other: "GraphTransformation") -> "GraphTransformation":
        """Composes transformations like functions.

        `composite = self*other` transforms `other.src_gm` to `self.mod_gm` by composing boundary maps.
        """
        assert (
            self.src_gm == other.mod_gm
        ), f"Mismatched contracting graph module for transformations {self} and {other}."
        return GraphTransformation(
            src_gm=other.src_gm,
            mod_gm=self.mod_gm,
            input_mod=lambda other_src_inputs: self.input_mod(
                other.input_mod(other_src_inputs)
            ),
            output_mod=lambda other_src_inputs, self_mod_outputs: (
                other.output_mod(
                    other_src_inputs,
                    self.output_mod(
                        other.input_mod(other_src_inputs),  # self_src_inputs
                        self_mod_outputs,
                    ),
                )
            ),
        )


def _get_io_from_gm(
    gm: GraphModule,
) -> tuple[list[Target], list[torch.Tensor | None]]:
    """Returns input nodes and output fake tensors from the graph module."""

    inputs: list[Target] = []
    meta_outputs: list[torch.Tensor | None] = []
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


def permute_metadata(
    source_node: Node, perms: tuple[Sequence[int] | None, ...]
) -> dict:
    """Returns a node meta resulting from applying `perm` to `source_node.meta`.

    If `source_node` returns multiple outputs, this will only return `val` metadata, and a tuple of permutations are expected.
    This is only handling `tensor_meta` and `val` entries, since only these are being used by the mlir import path.

    This function only supports nodes with at least one tensor output with a valid tensor meta dict.
    """

    def _permute_fake_tensor(
        og_val: torch.Tensor | None, perm: Sequence[int] | None
    ) -> torch.Tensor | None:
        """Some `None` og_vals are allowed here since some ops return `None`."""
        if perm is None or og_val is None:
            return og_val
        assert len(perm) == len(
            og_val.shape
        ), f"Invalid permutation, {perm} for value {og_val}."
        permuted_val = og_val.permute(*perm)
        return permuted_val

    def _permute_tensor_meta(
        og_meta: TensorMetadata, permuted_val: torch.Tensor
    ) -> TensorMetadata:
        if og_meta.is_quantized:
            raise NotImplementedError(
                f"Quantized layout handling NYI. Got meta {og_meta} for node {source_node}."
            )
        permuted_meta = TensorMetadata(
            shape=permuted_val.shape,
            dtype=permuted_val.dtype,
            requires_grad=og_meta.requires_grad,
            stride=permuted_val.stride(),
            memory_format=torch.contiguous_format,
            is_quantized=og_meta.is_quantized,
            qparams=og_meta.qparams,
        )
        return permuted_meta

    og_metas = source_node.meta.get("tensor_meta")
    og_vals = source_node.meta.get("val")

    # Nodes with multiple outputs will have no "tensor_meta" and a tuple "val".
    if isinstance(og_vals, tuple):
        assert (
            og_metas is None
        ), f"`tensor_meta` expected to be None for multi-ouptut node. Got {source_node.meta=}"
        assert isinstance(
            perms, tuple
        ), "Permutation for multi-output node must be tuple."
        new_vals: list[torch.Tensor | None] = []
        for og_val, perm in zip(og_vals, perms, strict=True):
            new_vals.append(_permute_fake_tensor(og_val, perm))
        return {"val": tuple(new_vals)}

    # Single output nodes are expected to have both "tensor_meta" and "val".
    # Nodes without output tensors are unexpected here.
    assert isinstance(
        og_metas, TensorMetadata
    ), f"Must have valid metadata, got metadata of type {type(og_metas)} for node {source_node}."
    assert len(perms) == 1, f"Expected one permutation, got {perms}."
    perm = perms[0]
    permuted_val = _permute_fake_tensor(og_vals, perm)
    assert isinstance(
        permuted_val, torch.Tensor
    ), f"Expected tensor value metadata for node {source_node}."
    permuted_meta = _permute_tensor_meta(og_metas, permuted_val)
    return {
        "tensor_meta": permuted_meta,
        "val": permuted_val,
    }


def call_permute(node: Node, perm: Sequence[int]) -> Node:
    """Returns the result of a permute operation as a node in the same fx graph.

    Does not attach metadata.
    """
    with node.graph.inserting_after(node):
        return node.graph.call_function(
            torch.ops.aten.permute.default, (node, list(perm))
        )


def convert_output(curr_output: Node) -> tuple[Node, MemoryFormatPermutation | None]:
    """Permutes the given node to contiguous format. Returns the created node and permutation used."""
    og_val = curr_output.meta["val"]
    perms = get_memory_format_permutation(og_val)
    if perms is None:
        return curr_output, perms
    new_out = call_permute(curr_output, perms.permutation)
    new_out.meta = permute_metadata(curr_output, (perms.permutation,))
    return new_out, perms


def convert_placeholder(
    src_pl: Node, target_graph: Graph
) -> tuple[Node, MemoryFormatPermutation | None]:
    perms = get_memory_format_permutation(src_pl.meta["val"])
    if perms is None:
        return target_graph.node_copy(src_pl), perms
    name = f"{src_pl.name}_boo"
    pl = target_graph.placeholder(name=name)
    pl.meta = permute_metadata(src_pl, (perms.permutation,))
    target_replacement = call_permute(pl, perms.inverse_permutation)
    target_replacement.meta = {k: v for k, v in src_pl.meta.items()}
    return target_replacement, perms


PermsTuple: TypeAlias = tuple[MemoryFormatPermutation | None, ...]


def get_graph_module_with_contiguous_boundary(
    src_gm: GraphModule,
) -> tuple[GraphModule, PermsTuple, PermsTuple]:
    """Returns a tuple containing:
    1. A GraphModule which applies `src_gm`, but contains contiguous boundary tensors.
    2. Memory format permutations used to force inputs to be contiguous.
    3. Memory format permutations used to force outputs to be contiguous.

    Returned permutations will be `None` whenever the corresponding boundary tensor is already contiguous.
    """
    src = src_gm.graph
    g = Graph()

    # Convert placeholders to contiguous placeholder + permute back to original format.
    src_placeholders = src.find_nodes(op="placeholder")
    val_map: dict[Node, Node] = {}
    input_perms: list[MemoryFormatPermutation | None] = []
    for src_pl in src_placeholders:
        pl_replacement, _perms = convert_placeholder(src_pl, g)
        val_map[src_pl] = pl_replacement
        input_perms.append(_perms)

    # Copy source graph body.
    output_og_args = g.graph_copy(src, val_map=val_map)

    # For boo.fusion, all subgraphs we extract return tuples of tensors.
    # E.g. single-output subgraphs should return `(output,)`.
    assert isinstance(output_og_args, tuple)
    # Convert outputs to contiguous format and collect perms.
    permuted_output_args: list[torch.fx.Node] = []
    output_perms: list[MemoryFormatPermutation | None] = []
    for ret in output_og_args:
        assert isinstance(
            ret, Node
        ), f"Expected returns to be nodes. Got {ret} with type {type(ret)}."
        new_ret, _perm = convert_output(ret)
        permuted_output_args.append(new_ret)
        output_perms.append(_perm)

    # Create an output node.
    g.create_node(op="output", target="output", args=(tuple(permuted_output_args),))

    # Make a GraphModule from the constructed Graph.
    new_gm = GraphModule(root=src_gm, graph=g)
    g.lint()
    new_gm.recompile()
    return new_gm, tuple(input_perms), tuple(output_perms)


def get_custom_graph_op(
    src_gm: GraphModule,
    *,
    force_single_dispatch: bool = False,
    inplace_convert: bool = True,
) -> torch._ops.OpOverloadPacket:
    """Converts a graph module into a custom operator.

    This function infers input/output signature from the graph metadata, and produces a specialized op.
    The returned op will not automatically re-specialize for different inputs.
    """
    gm, input_perms, output_perms = get_graph_module_with_contiguous_boundary(src_gm)
    gm_string = str(gm.print_readable(print_output=False, include_stride=True))
    hash = sha1(gm_string.encode(), usedforsecurity=False).hexdigest()
    call_function_names = "_".join(
        n.name[0:10]
        for n in gm.graph.nodes
        if n.op == "call_function"
        and not n.name.startswith("getitem")
        and not n.name.startswith("permute")
    )

    # Evidently, there is a limit to the number of characters in a path.
    # We use this name for the file cache, so some modest limits need to be set.
    # TODO: reorganize the file cache so this isn't problematic.
    op_name = (
        f"fused_op_{call_function_names}_{hash}"
        if len(call_function_names) < 120
        else f"fused_op_{hash}"
    )
    op_name += "_inplace" if inplace_convert else ""
    op_name = get_arg_spec_name(
        op_name, *[n.meta.get("val") for n in gm.graph.find_nodes(op="placeholder")]
    )
    logger.debug("Got hash str '%s' for GraphModule: \n %s", hash, gm_string)

    if not hasattr(torch.ops.boo, op_name):
        _define_custom_graph_op(
            gm,
            src_gm,
            op_name,
            input_perms,
            output_perms,
            force_single_dispatch=force_single_dispatch,
            inplace_convert=inplace_convert,
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


def _hack_inplace_exported_program(
    std_gm: GraphModule,
    output_mem_format_perms: Sequence[MemoryFormatPermutation | None],
) -> tuple[ExportedProgram, list[MemoryFormatPermutation | None], list[torch.Tensor]]:
    """Generates an `ExportedProgram` for a simple `GraphModule`, and converts output tensors to in-place mutations.

    The `torch.export` function isn't usable within `torch.compile`, and this function is a workaround to tell the
    `FxImporter` that we want inplace invocations.

    Some implicit assumptions:

    1. The graph modules we are encountering are stateless.
    2. All inputs/outputs are tensors with valid metadata.
    """
    std_g = std_gm.graph
    output_node = std_g.output_node()
    outs = output_node.args[0]
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    input_mutations: dict[str, torch.fx.Node] = {}
    seen_names: list[str] = []
    init_perms: list[MemoryFormatPermutation | None] = []
    init_fakes: list[torch.Tensor] = []
    for idx, o in enumerate(outs):
        assert isinstance(o, torch.fx.Node)
        with std_g.inserting_before():
            init_name = o.name + "_init"
            # Only create an init tensor once for repeated outputs.
            if init_name in seen_names:
                continue
            seen_names.append(init_name)
            init_plh = std_g.placeholder(name=init_name)
            v = o.meta.get("val")
            init_plh.meta = {
                "tensor_meta": o.meta.get("tensor_meta"),
                "val": o.meta.get("val"),
            }
            assert (
                isinstance(v, torch.Tensor) and v.is_contiguous()
            ), f"Expected only contiguous tensor outputs, got returned node {o} with metadata {o.meta}."
            init_perms.append(output_mem_format_perms[idx])
            init_fakes.append(v)
        with std_g.inserting_after(o):
            copy_node = std_g.call_function(
                torch.ops.aten.copy.default, args=(init_plh, o, True)
            )
            copy_node.meta = {k: v for k, v in init_plh.meta.items()}
            input_mutations[init_name] = copy_node

    new_outputs = tuple(input_mutations.values()) + tuple(outs)
    output_node.args = (new_outputs,) + tuple(output_node.args[1:])
    input_specs = [
        InputSpec(
            kind=InputKind.USER_INPUT,
            arg=TensorArgument(name=str(p.target)),
            target=None,
            persistent=None,
        )
        for p in std_g.find_nodes(op="placeholder")
    ]
    output_specs = list(
        OutputSpec(
            kind=OutputKind.USER_INPUT_MUTATION,
            arg=TensorArgument(name=o_init.name),
            target=key,
        )
        for key, o_init in input_mutations.items()
    )
    output_specs.extend(
        [
            OutputSpec(
                kind=OutputKind.USER_OUTPUT,
                arg=TensorArgument(name=o.name),
                target=None,
            )
            for o in outs
        ]
    )

    fake_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )
    std_g.eliminate_dead_code()
    std_g.lint()
    std_gm.recompile()

    fake_ep = ExportedProgram(
        root=std_gm,
        graph=std_gm.graph,
        graph_signature=fake_graph_signature,
        state_dict={},
        range_constraints={},
        module_call_graph=[],
        example_inputs=tuple(
            n.meta["val"] for n in std_gm.graph.find_nodes(op="placeholder")
        ),
    )
    return fake_ep, init_perms, init_fakes


def _define_custom_graph_op(
    gm: GraphModule,
    og_gm: GraphModule,
    op_name: str,
    input_mem_format_perms: Sequence[MemoryFormatPermutation | None],
    output_mem_format_perms: Sequence[MemoryFormatPermutation | None],
    *,
    force_single_dispatch: bool = False,
    inplace_convert: bool = False,
):
    """Defines a custom op from the graph module with given op_name in the boo library."""
    inputs, outputs = _get_io_from_gm(gm)
    is_none_output = _maybe_trim_none_outputs(gm)
    has_a_none_output = any(is_none_output)
    schema = _get_schema(inputs, outputs)
    define_schema(op_name, schema)
    init_fakes: list[torch.Tensor] = []
    if inplace_convert:
        (program, init_perms, init_fakes) = _hack_inplace_exported_program(
            gm, output_mem_format_perms
        )
    else:
        program = gm

    @register_impl(op_name)
    def _(*args):
        handled_inputs = _handle_layouts(
            args, perms=input_mem_format_perms, perm_item="permutation"
        )
        if inplace_convert:
            _device = lambda fake: args[0].device if len(args) > 0 else fake.device
            init_tensors = tuple(
                torch.empty(*fake.shape, dtype=fake.dtype, device=_device(fake))
                for fake in reversed(init_fakes)
            )
            handled_inputs = init_tensors + handled_inputs
        # TODO: hoist this out of runtime once we remove the last torch.export from conv custom ops.
        launch = get_launchable_from_traced_object(
            program=program,
            func_name=op_name,
            force_single_dispatch=force_single_dispatch,
        )
        outputs = launch(*[arg.data for arg in handled_inputs])
        single_output = False
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
            single_output = True
        if outputs is None:
            outputs = tuple()
        assert isinstance(
            outputs, (list, tuple)
        ), f"Got outputs {outputs} of unhandled type {type(outputs)}."
        assert len(outputs) == len(output_mem_format_perms)
        handled_outputs = _handle_layouts(
            outputs, perms=output_mem_format_perms, perm_item="inverse_permutation"
        )
        if not has_a_none_output:
            return handled_outputs[0] if single_output else handled_outputs
        # We have at least one None output that needs to be included.
        # Handle this better.
        all_results: list[torch.Tensor | None] = []
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
        outputs = og_gm.forward(*args)
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
