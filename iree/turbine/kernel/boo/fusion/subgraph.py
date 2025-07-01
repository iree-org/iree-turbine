# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence, Set, List, Dict, Tuple

import torch
from torch.fx.subgraph_rewriter import replace_pattern
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch._functorch.aot_autograd import _aot_export_function

from .schema import FusionSchema, OpFusionSpec, Target
from ..ops.graph import get_autograd_function
from ....support.logging import aot_logger as logger


def _log_graph_module(label: str, gm: GraphModule):
    logger.debug("%s:\n%s", label, gm.print_readable(print_output=False))


def get_subgraph_replacement(subgraph: GraphModule):
    _log_graph_module("Extracted SubGraph Module", subgraph)
    fake_args = tuple(
        [n.meta.get("val") for n in subgraph.graph.nodes if n.op == "placeholder"]
    )
    joint_sg, metadata, in_spec, out_spec = _aot_export_function(
        subgraph.forward,
        fake_args,
        decompositions=None,
    )
    # TODO: do some minimal validation on the results of the above function.
    # in_spec, _kw_in_spec = in_spec.children_specs
    _log_graph_module("AOT Joint FWD/BWD Subgraph Module", joint_sg)

    fake_args_joint = tuple(
        [n.meta.get("val") for n in joint_sg.graph.nodes if n.op == "placeholder"]
    )

    custom_op = get_autograd_function(
        joint_sg,
        fake_args_joint,
        num_fwd_outputs=metadata.num_forward_returns,
        force_single_dispatch=False,
    )

    single_node_graph = fused_subgraph(
        subgraph,
        custom_op,
        (n for n in subgraph.graph.nodes if n.op == "placeholder"),
        num_outputs=metadata.num_forward_returns,
    )
    _log_graph_module("Replacement Subgraph", single_node_graph)
    return single_node_graph


def fused_subgraph(
    root, target: Target, input_nodes: Sequence[Node], num_outputs: int
) -> GraphModule:
    g = Graph()
    new_inputs = []
    for n in input_nodes:
        new_node = g.placeholder(n.name)
        new_node.meta = n.meta
        new_inputs.append(new_node)
    fused_node = g.call_function(target, tuple(new_inputs))
    outputs = []
    if num_outputs == 1:
        g.output(result=fused_node)
        return GraphModule(root, g)
    for i in range(num_outputs):
        outputs.append(g.call_method("__getitem__", args=(fused_node, i)))
    g.output(result=tuple(outputs))
    return GraphModule(root, g)


def extract_fusion_subgraph_modules(
    src_gm: GraphModule, fusion_schema: FusionSchema
) -> Tuple[Sequence[GraphModule], Sequence[Dict[Node, Node]]]:
    """Traverses src_gm nodes in order. When a node matches a root op in the fusion_schema,
    A new subgraph is created with the root op in addition to any adjacent nodes matching the schema.

    All subgraphs found this way are returned in a list.
    The corresponding mappings from src_gm nodes -> subgraph nodes is also returned as a list.
    """
    subgraph_projections: List[Dict[Node, Node]] = []
    subgraph_inclusions: List[Dict[Node, Node]] = []
    subgraphs: List[GraphModule] = []
    used_nodes: Set[Node] = set()
    for root in src_gm.graph.nodes:
        # if this node is included in any subgraph already, we can't use it.
        if root in used_nodes:
            continue

        # TODO: support other situations, e.g. "call_method"
        if root.op != "call_function":
            continue
        # Check if this node is a fusion root op
        node_spec: OpFusionSpec | None = fusion_schema.get(root.target, None)
        if node_spec is None:
            continue
        node_list = [root]

        # Walk producers from root and include them in the subgraph
        worklist = [root]
        while len(worklist) > 0:
            # We are treating worklist as a FIFO queue (pop from front).
            # This results in a breadth-first traversal of producers.
            curr_node = worklist.pop(0)
            for producer in curr_node.all_input_nodes:
                if producer.op != "call_function":
                    continue
                if producer.target not in node_spec.producers:
                    continue
                if producer in used_nodes:
                    continue
                # Insert producers at the front, since we want to preserve at least some weak ordering of nodes.
                # Is it possible for this to generate an invalid ordering? (Maybe it's better to just sort node_list after).
                node_list.insert(0, producer)
                if node_spec.recursive:
                    worklist.append(producer)

        # Walk consumers from root and include them in the subgraph
        worklist = [root]
        while len(worklist) > 0:
            curr_node = worklist.pop(0)
            for consumer in curr_node.users:
                if consumer.op != "call_function":
                    continue
                if consumer.target not in node_spec.consumers:
                    continue
                if consumer in used_nodes:
                    continue
                node_list.append(consumer)
                if node_spec.recursive:
                    worklist.append(consumer)

        # Create a detached subgraph
        subgraph = Graph()
        output_nodes = []
        subgraph_projection = {}
        subgraph_inclusion = {}
        for node in node_list:
            # Iterate over producers in src graph and make placeholders in detached subgraph if necessary.
            for producer in node.all_input_nodes:
                if producer in node_list:
                    continue
                subgraph_input = subgraph.placeholder(name=producer.name)
                subgraph_input.meta = producer.meta
                tensor_meta = producer.meta.get("tensor_meta")
                tensor_val = producer.meta.get("val")
                # Workaround to indicate that intermediate inputs require gradient calculation
                if (
                    tensor_meta is not None
                    and isinstance(tensor_val, torch.Tensor)
                    and tensor_meta.requires_grad
                ):
                    tensor_val.requires_grad = True
                subgraph_projection[producer] = subgraph_input
                subgraph_inclusion[subgraph_input] = producer
            # Copy over the current node to the detached subgraph, updating args based on corresponding elements of the detached subgraph.
            new_node = subgraph.node_copy(node, arg_transform=subgraph_projection.get)
            subgraph_projection[node] = new_node
            subgraph_inclusion[new_node] = node
            # Any nodes in the subgraph which have users outside the subgraph should return.
            if len(set(node.users.keys()).difference(node_list)) > 0:
                output_nodes.append(new_node)

        # Add outputs to the detached subgraph.
        subgraph.output(result=tuple(output_nodes))
        subgraph.lint()

        subgraph_projections.append(subgraph_projection)
        subgraph_inclusions.append(subgraph_inclusion)
        used_nodes.update(subgraph_projection.keys())
        subgraphs.append(GraphModule(src_gm, subgraph))

    return subgraphs, subgraph_projections


def replace_subgraphs(src_gm, external_subgraphs, replacements):
    """Makes a copy of src_gm and replaces instances of each subgraph with their corresponding replacement graph."""
    for sg, replacement in zip(external_subgraphs, replacements):
        _ = replace_pattern(src_gm, sg, replacement)
    return src_gm
