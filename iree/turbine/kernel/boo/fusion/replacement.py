from collections.abc import Callable

import torch
from torch import fx
from torch.fx.node import Argument, Target


from iree.turbine.kernel.boo import ops as boo_ops

# 1 to 1 op replacements. The replacement function should return the new target
# and arguments, or 'None' if it doesn't apply.
ReplacementSchema = dict[
    Target,
    Callable[
        [tuple[Argument, ...], dict[str, object]],  # (node.args, node.meta)
        tuple[Callable, tuple[Argument, ...]] | None,  # (new_target, new_args) | None
    ],
]


def apply_replacements(graph: fx.Graph, replacements: ReplacementSchema):
    for node in graph.nodes:
        if node.op == "call_function":
            replacer_fn = replacements.get(node.target, lambda *_: None)
            replacement = replacer_fn(node.args, node.meta)
            if replacement is None:
                continue
            target, target_args = replacement
            with graph.inserting_after(node):
                call_boo = graph.call_function(target, target_args)
            node.replace_all_uses_with(call_boo, propagate_meta=True)
            graph.erase_node(node)


def replace_aten_convolution(args: tuple[Argument, ...], meta: dict[str, object]):
    "Replace 'torch.ops.aten.convolution' with custom BOO implementation."
    (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        groups,
    ) = args

    # BOO convolution doesn't support transpose. 'output_padding' is ignored
    # in non-transpose cases.
    if transposed is not False:
        return None

    example_out = meta["val"]
    assert isinstance(example_out, torch.Tensor)
    output_is_channels_last = example_out.is_contiguous(
        memory_format=torch.channels_last
    )
    return boo_ops.convolution_replacement, (
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        output_is_channels_last,
    )


def replace_aten_convolution_backward(
    args: tuple[Argument, ...], meta: dict[str, object]
) -> tuple[Callable, tuple[Argument, ...]] | None:
    "Replace 'torch.ops.aten.convolution' with custom BOO implementation."
    (
        grad_output,
        input,
        weight,
        _bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        _output_padding,
        groups,
        output_mask,
    ) = args

    if transposed is not False:
        return None

    return boo_ops.convolution_backward_replacement, (
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        groups,
        output_mask,
    )
