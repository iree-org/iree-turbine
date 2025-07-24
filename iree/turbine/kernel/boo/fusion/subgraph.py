# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import NamedTuple

import torch
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from .schema import FusionSchema
from ....support.logging import aot_logger as logger


class FusedSubgraph(NamedTuple):
    module: GraphModule
    """Module containing the subgraph to be fused."""
    single_dispatch: bool
    """Whether to force compile the subgraph as a single dispatch."""
    matched_nodes: list[Node]
    """Nodes in the orignal graph that were matched."""
    arguments: list[Node]
    """Arguments that should be passed to 'module' when calling it."""
    results: list[Node]
    """Nodes in the original graph that are produced by calling 'module'."""


def extract_fusion_subgraph_modules(
    src_gm: GraphModule, fusion_schema: FusionSchema
) -> list[FusedSubgraph]:
    """Traverses src_gm nodes in order. When a node matches a root op in the fusion_schema,
    A new subgraph is created with the root op in addition to any adjacent nodes matching the schema.

    All subgraphs found this way are returned in a list.
    The corresponding mappings from src_gm nodes -> subgraph nodes is also returned as a list.
    """
    subgraphs: list[FusedSubgraph] = []
    used_nodes: set[Node] = set()
    for root in src_gm.graph.nodes:
        # if this node is included in any subgraph already, we can't use it.
        if root in used_nodes:
            continue

        # TODO: support other situations, e.g. "call_method"
        if root.op != "call_function":
            continue
        # Check if this node is a fusion root op
        node_spec = fusion_schema.get(root.target, None)
        if node_spec is None:
            continue

        if not node_spec.check_filters(root):
            continue

        node_list = [root]

        # Walk producers from root and include them in the subgraph
        worklist = [root]
        visited_nodes: set = {root}
        while len(worklist) > 0:
            # We are treating worklist as a FIFO queue (pop from front).
            # This results in a breadth-first traversal of producers.
            curr_node = worklist.pop(0)
            for producer in curr_node.all_input_nodes:
                if producer in visited_nodes:
                    continue
                if producer in used_nodes:
                    continue
                if not node_spec.is_fusable_producer(producer):
                    continue
                # Insert producers at the front, since we want to preserve at least some weak ordering of nodes.
                # Is it possible for this to generate an invalid ordering? (Maybe it's better to just sort node_list after).
                visited_nodes.add(producer)
                node_list.insert(0, producer)
                if node_spec.recursive:
                    worklist.append(producer)

        # Walk consumers from root and include them in the subgraph
        worklist = [root]
        visited_nodes: set = {root}
        while len(worklist) > 0:
            curr_node = worklist.pop(0)
            for consumer in curr_node.users:
                if consumer in visited_nodes:
                    continue
                if consumer in used_nodes:
                    continue
                if not node_spec.is_fusable_consumer(consumer):
                    continue
                visited_nodes.add(consumer)
                node_list.append(consumer)
                if node_spec.recursive:
                    worklist.append(consumer)

        logger.debug("pre-sort node_list: %s", str(node_list))
        node_list = sorted(node_list)
        logger.debug("post-sort node_list: %s", str(node_list))
        # Create a detached subgraph
        subgraph = Graph()
        output_nodes: list[Node] = []
        subgraph_matched_nodes: list[Node] = []
        subgraph_arguments: list[Node] = []
        subgraph_results: list[Node] = []
        subgraph_projection: dict[Node, Node] = {}
        for node in node_list:
            # Iterate over producers in src graph and make placeholders in detached subgraph if necessary.
            for producer in node.all_input_nodes:
                if producer in node_list or producer in subgraph_projection.keys():
                    continue
                subgraph_arguments.append(producer)
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
            # Copy over the current node to the detached subgraph, updating args based on corresponding elements of the detached subgraph.
            new_node = subgraph.node_copy(node, arg_transform=subgraph_projection.get)
            subgraph_projection[node] = new_node
            subgraph_matched_nodes.append(node)
            # Any nodes in the subgraph which have users outside the subgraph should be returned.
            if len(set(node.users.keys()).difference(node_list)) > 0:
                output_nodes.append(new_node)
                subgraph_results.append(node)

        # Add outputs to the detached subgraph.
        subgraph.output(result=tuple(output_nodes))
        subgraph.lint()

        used_nodes.update(subgraph_projection.keys())
        subgraphs.append(
            FusedSubgraph(
                module=GraphModule(src_gm, subgraph),
                single_dispatch=node_spec.make_single_dispatch,
                matched_nodes=subgraph_matched_nodes,
                arguments=subgraph_arguments,
                results=subgraph_results,
            )
        )

    return subgraphs
