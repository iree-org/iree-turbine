import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List, Optional, Mapping

from .decompositions import DEFAULT_DECOMPOSITIONS


def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: Optional[List[torch._ops.OpOverload]] = None,
):
    if decompose_ops is None:
        return gm

    decompositions: Mapping = get_decompositions(decompose_ops)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)  # type: ignore
