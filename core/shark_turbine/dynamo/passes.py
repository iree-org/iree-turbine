import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from shark_turbine.dynamo import utils
from torch.func import functionalize
from typing import List

# default decompositions pulled from SHARK / torch._decomp
DEFAULT_DECOMPOSITIONS = [
    torch.ops.aten.embedding_dense_backward,
    torch.ops.aten.native_layer_norm_backward,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.norm.ScalarOpt_dim,
    torch.ops.aten.native_group_norm,
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes,
    torch.ops.aten.native_layer_norm,
    torch.ops.aten.masked_fill.Tensor,
    torch.ops.aten.masked_fill.Scalar,
    torch.ops.aten.t,
    torch.ops.aten.addmm,
    # decompositions that aid us in handling nn.BatchNorm2d
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit_no_training,
    torch.ops.aten._native_batch_norm_legit,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten.squeeze.dims,
    # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
    torch.ops.aten.soft_margin_loss,
    torch.ops.aten.im2col,
    torch.ops.aten._euclidean_dist,
    torch.ops.aten.index_copy,
    torch.ops.aten.index_copy_,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.log_sigmoid_forward,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.binary_cross_entropy,
    torch.ops.aten.dot,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._prelu_kernel,
    torch.ops.aten.full,
    torch.ops.aten._log_softmax,
    torch.ops.aten.nll_loss_forward,
    torch.ops.aten.nll_loss_backward,
    torch.ops.aten._to_copy,
    torch.ops.aten._log_softmax_backward_data,
    torch.ops.aten.lift_fresh_copy.default,
    torch.ops.aten._unsafe_index.Tensor,
    torch.ops.aten.unbind.int,
    # decompositions added manually in this file
    torch.ops.aten._scaled_dot_product_flash_attention.default,
]


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: List[torch._ops.OpOverload] = None,
):
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)
