from collections.abc import Callable
from typing import Literal

from operator import getitem
import torch
from torch import fx
from torch.fx.node import Target, Node

from ..op_exports.conv import DEFAULT_LAYOUTS
from iree.turbine.kernel.boo import ops as boo_ops
from ..ops.utils import get_memory_format_permutation, MemoryFormatPermutation
from ..ops.graph import call_permute, permute_metadata

# 1 to 1 op replacements. The replacement function should modify input Node graph if applicable.
ReplacementSchema = dict[Target, Callable[[Node], None]]


def apply_replacements(graph: fx.Graph, replacements: ReplacementSchema):
    for node in graph.nodes:
        if node.op == "call_function":
            replacer_fn = replacements.get(node.target, lambda n: None)
            replacer_fn(node)


def _apply_perms(
    src_node: Node,
    perms: MemoryFormatPermutation | None,
    perm_item: Literal["permutation", "inverse_permutation"],
) -> Node:
    if perms is None:
        return src_node
    new = call_permute(src_node, getattr(perms, perm_item))
    new.meta = permute_metadata(src_node, getattr(perms, perm_item))
    return new


def replace_aten_convolution(node: Node):
    "Replace 'torch.ops.aten.convolution' with custom BOO implementation."
    (
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        groups,
    ) = node.args

    if transposed is not False:
        return

    graph = node.graph

    assert isinstance(x, Node)
    x_fake = x.meta.get("val")
    assert isinstance(x_fake, torch.Tensor)
    assert isinstance(w, Node)
    w_fake = w.meta.get("val")
    assert isinstance(w_fake, torch.Tensor)
    output_fake = node.meta.get("val")
    assert isinstance(output_fake, torch.Tensor)

    num_spatial_dims = len(x_fake.shape) - 2
    pytorch_layout = DEFAULT_LAYOUTS[num_spatial_dims]

    x_perms = get_memory_format_permutation(x_fake, num_spatial_dims)
    w_perms = get_memory_format_permutation(w_fake, num_spatial_dims)
    output_perms = get_memory_format_permutation(output_fake, num_spatial_dims)

    x_contig = _apply_perms(x, x_perms, "permutation")
    w_contig = _apply_perms(w, w_perms, "permutation")

    to_layout = lambda perms: (
        pytorch_layout
        if perms is None
        else "".join([pytorch_layout[p] for p in perms.permutation])
    )

    call_args = (
        x_contig,
        w_contig,
        b,
        stride,
        padding,
        dilation,
        groups,
        to_layout(x_perms),
        to_layout(w_perms),
        to_layout(output_perms),
    )
    with graph.inserting_before(node):
        replacement_conv = graph.call_function(
            torch.ops.boo.layout_customizable_convolution.default, args=call_args
        )
        replacement_conv.meta = (
            node.meta
            if output_perms is None
            else permute_metadata(node, output_perms.permutation)
        )
        post_permute = _apply_perms(
            replacement_conv, output_perms, "inverse_permutation"
        )

    node.replace_all_uses_with(replace_with=post_permute)
    graph.erase_node(node)
    graph.lint()


def replace_aten_convolution_backward(node: Node):
    "Replace 'torch.ops.aten.convolution_backward' with custom BOO implementation."
    (
        grad_output,
        x,
        w,
        _bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        groups,
        output_mask,
    ) = node.args

    if transposed is not False:
        return

    graph = node.graph

    assert isinstance(x, Node)
    x_fake = x.meta.get("val")
    assert isinstance(x_fake, torch.Tensor)
    assert isinstance(w, Node)
    w_fake = w.meta.get("val")
    assert isinstance(w_fake, torch.Tensor)
    assert isinstance(grad_output, Node)
    output_fake = grad_output.meta.get("val")
    assert isinstance(output_fake, torch.Tensor)

    num_spatial_dims = len(x_fake.shape) - 2
    pytorch_layout = DEFAULT_LAYOUTS[num_spatial_dims]

    x_perms = get_memory_format_permutation(x_fake, num_spatial_dims)
    w_perms = get_memory_format_permutation(w_fake, num_spatial_dims)
    output_perms = get_memory_format_permutation(output_fake, num_spatial_dims)

    x_contig = _apply_perms(x, x_perms, "permutation")
    w_contig = _apply_perms(w, w_perms, "permutation")
    grad_output_contig = _apply_perms(grad_output, output_perms, "permutation")

    to_layout = lambda perms: (
        pytorch_layout
        if perms is None
        else "".join([pytorch_layout[p] for p in perms.permutation])
    )

    call_args = (
        grad_output_contig,
        x_contig,
        w_contig,
        stride,
        padding,
        dilation,
        groups,
        to_layout(x_perms),
        to_layout(w_perms),
        to_layout(output_perms),
        output_mask,
    )
    ret_grad_perms = (x_perms, w_perms, None)
    with graph.inserting_before(node):
        replacement_conv = graph.call_function(
            torch.ops.boo.layout_customizable_convolution_backward.default,
            args=call_args,
        )
        replacement_conv.meta = node.meta
        old_val = node.meta.get("val")
        assert (
            isinstance(old_val, tuple) and len(old_val) == 3
        ), f'Invalid metadata for backward conv. Got node.meta["val"] = {old_val}.'
        new_val = tuple(
            (
                val
                if val is None or perms is None
                else permute_metadata(val, perms.inverse_permutation)
            )
            for val, perms in zip(old_val, ret_grad_perms)
        )
        replacement_conv.meta["val"] = new_val
    original_users = list(node.users.keys())
    for consumer in original_users:
        assert consumer.op == "call_function"
        assert consumer.target == getitem
        assert isinstance(consumer.args, tuple) and len(consumer.args) == 2
        index = consumer.args[-1]
        assert isinstance(index, int) and index in range(3)
        _perm = ret_grad_perms[index]
        with graph.inserting_before(consumer):
            new_output = graph.call_function(getitem, args=(replacement_conv, index))
            new_output.meta = (
                consumer.meta
                if _perm is None
                else permute_metadata(consumer, _perm.permutation)
            )
            replacement = _apply_perms(new_output, _perm, "inverse_permutation")
            consumer.replace_all_uses_with(replace_with=replacement)
            graph.erase_node(consumer)

    graph.erase_node(node)
    graph.lint()
