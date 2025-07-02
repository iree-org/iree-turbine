import torch

from dataclasses import dataclass
from typing import Optional, List
import math

from ..exports.signature import OpSignature


@dataclass
class GEMMSignature(OpSignature):
    """Signature for General Matrix Multiplication (GEMM) operations"""

    a_shape: List[int]
    b_shape: List[int]
    transpose_a: bool = False
    transpose_b: bool = False
    dtype: torch.dtype = torch.float32

    def get_func_name(self) -> str:
        """Format: gemm_{dtype}_{m}x{k}x{n}[_transA][_transB]"""
        m, k = self.a_shape
        _, n = self.b_shape

        name_items = [
            "gemm",
            str(self.dtype).removeprefix("torch."),
            f"{m}x{k}x{n}",
        ]
        if self.transpose_a:
            name_items.append("_transA")
        if self.transpose_a:
            name_items.append("_transB")
        return "_".join(name_items)

    def get_output_size(self) -> int:
        """Returns the size of the output tensor in bytes."""
        m = self.a_shape[0]
        n = self.b_shape[1]
        return math.prod([m, n]) * self.dtype.itemsize

    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        class Linear(torch.nn.Module):
            def __init__(self, sig: GEMMSignature):
                super().__init__()
                self.sig = sig

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                if self.sig.transpose_a:
                    a = a.t()
                if self.sig.transpose_b:
                    b = b.t()
                return torch.mm(a, b)

        return Linear(self)

    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen = gen.manual_seed(seed)

        def get(shape: List[int]) -> torch.Tensor:
            if splat_value is not None:
                return torch.ones(shape, dtype=self.dtype, device=device) * splat_value
            return torch.randn(shape, generator=gen, dtype=self.dtype, device=device)

        a_shape = self.a_shape
        if self.transpose_a:
            a_shape = [self.a_shape[1], self.a_shape[0]]
        b_shape = self.b_shape
        if self.transpose_b:
            b_shape = [self.b_shape[1], self.b_shape[0]]

        return (get(a_shape), get(b_shape))
