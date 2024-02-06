import torch
from torch._prims_common.wrappers import out_wrapper
from torch._prims_common import (
    DeviceLikeType,
    TensorLikeType,
)
import torch._refs as _refs
from torch._decomp import get_decompositions, register_decomposition
from torch import Tensor
from typing import Dict, List, Tuple, Optional


if torch.__version__ < "2.2.0":
    # Torch versions prior to 2.2.0 lacked some decompositions, which we
    # add manually.
    @register_decomposition(torch.ops.aten._scaled_dot_product_flash_attention.default)
    def scaled_dot_product_flash_attention(
        query,
        key,
        value,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        return_debug_mask: bool = False,
        *,
        scale: float = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, Tensor, Tensor, Tensor]:
        dtype = query.dtype
        batchSize, num_head, qSize, headSize = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        )

        logsumexp = torch.empty(
            [batchSize, qSize, num_head, headSize], dtype=torch.float
        )
        cum_seq_q, cum_seq_k = torch.empty([], dtype=torch.long), torch.empty(
            [], dtype=torch.long
        )
        max_q, max_k = 0, 0
        philox_seed, philox_offset = torch.empty([], dtype=torch.long), torch.empty(
            [], dtype=torch.long
        )
        debug_attn_mask = torch.empty(
            [],
            dtype=query.dtype,
            device="cpu",
            requires_grad=query.requires_grad,
        )
        output, _ = torch.ops.aten._scaled_dot_product_attention_math.default(
            query, key, value, None, dropout_p, is_causal, None, scale=scale
        )
        output = output.transpose(1, 2).contiguous(
            memory_format=torch.contiguous_format
        )
        return (
            output.transpose(1, 2),
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        )


# manually add decomposition to bypass the error that comes
# from VAE encode(inp).latent_dist.sample() failing to symbolically
# trace from torch fx.
# Expected Torch stable version: > 2.1.0
# diffusers side issue: https://github.com/huggingface/diffusers/issues/6239
# temporary Torch fix: https://github.com/pytorch/pytorch/issues/107170
@register_decomposition(torch.ops.aten.randn.generator)
@out_wrapper()
def randn_generator(
    *shape,
    generator: Optional[torch.Generator] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[DeviceLikeType] = None,
    layout: Optional[torch.layout] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> TensorLikeType:
    # We should eventually support the generator overload.
    # However, if someone passes in a None generator explicitly,
    # we can jut fall back to randn.default
    if generator is None:
        return _refs.randn(
            *shape,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
            pin_memory=pin_memory,
        )
    return NotImplemented
