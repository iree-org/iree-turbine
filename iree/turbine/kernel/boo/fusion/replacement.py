from collections.abc import Callable
from typing import Sequence

from operator import getitem
import torch
from torch import fx
from torch.fx.node import Target, Node

from ..op_exports.conv import DEFAULT_LAYOUTS
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
    forward_perm: bool,
) -> Node:
    """
    Does `call_permute` on `src_node` and handles newly created node metadata.

    If `forward_perm=True`, this will use `perms.permutation`. Otherwise, this
    will use `perms.inverse_permutation`.
    """
    if perms is None:
        return src_node
    p = perms.permutation if forward_perm else perms.inverse_permutation
    new = call_permute(src_node, p)
    new.meta = permute_metadata(src_node, (p,))
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

    x_contig = _apply_perms(x, x_perms, forward_perm=True)
    w_contig = _apply_perms(w, w_perms, forward_perm=True)

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
            else permute_metadata(node, (output_perms.permutation,))
        )
        post_permute = _apply_perms(replacement_conv, output_perms, forward_perm=False)

    node.replace_all_uses_with(replace_with=post_permute)
    graph.erase_node(node)
    graph.lint()


def replace_getitem_users(
    og_node: Node,
    replacement_node: Node,
    permutations: Sequence[MemoryFormatPermutation | None],
):
    """This is a helper function for handling multi-output node replacements.

    E.g., if `og_node` has three tensor outputs, typically we expect the only users of this node to be `getitem` users.

    Each item of the output tuple for `replacement_node` may have a shape that is a permutation of the corresponding output shape from `og_node`.
    The argument `permutations` should be a Sequence of `MemoryFormatPermutation | None` which would permute `og_node` output shape
    to `replacement_node` output shape. If `None` is passed for a permutation, it is expected these outputs have the same shape and strides.

    This function assumes:
        1. `og_node` and `replacement_node` are multi-output nodes with the same number of output tensors.
        2. `og_node` and `replacement_node` are members of the same graph.
        3. The only users of `og_node` are `getitem` nodes.

    Each `getitem(og_node, i)` will be replaced with `permute(getitem(replacement_node, i), permutations[i].inverse_permutation)`.
    """
    graph = og_node.graph
    if replacement_node.graph != graph:
        raise ValueError("Nodes must be members of the same graph.")
    original_users = list(og_node.users.keys())
    for use in original_users:
        assert (
            use.op == "call_function" and use.target == getitem
        ), f"This function assumes all users are `getitem` ops for the given multi-output node, {og_node}."
        assert (
            isinstance(use.args, tuple) and len(use.args) == 2
        ), f"Node `getitem` should have args=(multi-output-op, index), got {use.args}."
        index = use.args[-1]
        assert isinstance(index, int) and index in range(len(permutations))
        _perm = permutations[index]
        with graph.inserting_before(use):
            new_output = graph.call_function(getitem, args=(replacement_node, index))
            new_output.meta = (
                use.meta
                if _perm is None
                else permute_metadata(use, (_perm.permutation,))
            )
            replacement = _apply_perms(new_output, _perm, forward_perm=False)
            use.replace_all_uses_with(replace_with=replacement)
            graph.erase_node(use)


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

    x_contig = _apply_perms(x, x_perms, forward_perm=True)
    w_contig = _apply_perms(w, w_perms, forward_perm=True)
    grad_output_contig = _apply_perms(grad_output, output_perms, forward_perm=True)

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

    forward_perms = tuple(
        (None if p is None else p.permutation) for p in ret_grad_perms
    )
    replacement_conv.meta = permute_metadata(node, forward_perms)

    replace_getitem_users(node, replacement_conv, ret_grad_perms)
    graph.erase_node(node)
    graph.lint()


def _replace_sdpa_variant(
    node: Node,
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
):
    """Common helper to replace SDPA variants with torch.ops.aten.scaled_dot_product_attention.

    This handles the common logic of replacing a multi-output SDPA variant (flash or efficient)
    with the single-output scaled_dot_product_attention op.
    """
    # These SDPA variants return a tuple (output, ...).
    # We need to replace getitem(node, 0) with the replacement output
    # and remove other getitem users.
    users_to_process = list(
        [
            user
            for user in list(node.users.keys())
            if user.op == "call_function" and user.target == getitem
        ]
    )
    user_to_replace = None
    for user in users_to_process:
        assert isinstance(user.args, tuple) and len(user.args) == 2
        if user.args[1] != 0 and len(user.users) > 0:
            # Auxiliary output is being used downstream - cannot replace
            return
        elif user.args[1] == 0:
            user_to_replace = user

    graph = node.graph

    # enable_gqa is not forwarded through flash_attention, so we can drop it from kwargs.
    new_kwargs = {"scale": scale}

    # Insert replacement call before the original node.
    with graph.inserting_before(node):
        replacement = graph.call_function(
            torch.ops.aten.scaled_dot_product_attention.default,
            args=(query, key, value, attn_mask, dropout_p, is_causal),
            kwargs=new_kwargs,
        )
    user_to_replace.replace_all_uses_with(replacement, propagate_meta=True)
    for user in users_to_process:
        graph.erase_node(user)

    graph.erase_node(node)
    graph.lint()


def replace_aten_scaled_dot_product_flash_attention(node: Node):
    """Replace 'torch.ops.aten._scaled_dot_product_flash_attention' with 'torch.ops.aten.scaled_dot_product_attention'.

    Flash attention returns a tuple (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
    philox_seed, philox_offset, debug_attn_mask). For inference without dropout, only the
    output tensor (index 0) is needed.

    This replacement only supports cases where the auxiliary outputs are unused. If any
    output other than index 0 has downstream consumers, we bail out without modifying the graph.

    Flash attention signature:
        _scaled_dot_product_flash_attention(query, key, value, dropout_p, is_causal, *, scale=None)
    """
    # Extract arguments from flash attention call.
    # Flash attention signature: (query, key, value, dropout_p, is_causal, *, scale=None)
    query, key, value, dropout_p, is_causal = (
        node.args[0],
        node.args[1],
        node.args[2],
        node.args[3],
        node.args[4],
    )
    scale = node.kwargs.get("scale", None)

    # Attn mask is not present in the flash_attn op; pass None.
    _replace_sdpa_variant(node, query, key, value, None, dropout_p, is_causal, scale)


def replace_aten_scaled_dot_product_efficient_attention(node: Node):
    """Replace 'torch.ops.aten._scaled_dot_product_efficient_attention' with 'torch.ops.aten.scaled_dot_product_attention'.

    Efficient attention returns a tuple (output, log_sumexp, philox_seed, philox_offset).
    For inference without dropout, only the output tensor (index 0) is needed.

    This replacement only supports cases where the auxiliary outputs are unused. If any
    output other than index 0 has downstream consumers, we bail out without modifying the graph.

    Efficient attention signature:
        _scaled_dot_product_efficient_attention(query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, *, scale=None)
    """
    # Extract arguments from efficient attention call.
    # Efficient attention signature: (query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, *, scale=None)
    query, key, value, attn_bias, _compute_log_sumexp, dropout_p, is_causal = (
        node.args[0],
        node.args[1],
        node.args[2],
        node.args[3],
        node.args[4],
        node.args[5],
        node.args[6],
    )
    scale = node.kwargs.get("scale", None)

    _replace_sdpa_variant(
        node, query, key, value, attn_bias, dropout_p, is_causal, scale
    )
