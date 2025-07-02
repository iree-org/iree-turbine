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
    dtype: torch.dtype = torch.float32

    @property
    def output_shape(self) -> list[int]:
        """Returns the output shape (M, N) for the GEMM operation"""
        m = self.a_shape[0]
        n = self.b_shape[1]
        return [m, n]

    @staticmethod
    def get(
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> "GEMMSignature":
        """Create a GEMMSignature from input tensors."""
        # Validate shapes
        assert len(a.shape) == 2, f"Expected 2D matrix for A, got shape {a.shape}"
        assert len(b.shape) == 2, f"Expected 2D matrix for B, got shape {b.shape}"
        assert a.shape[1] == b.shape[0], (
            f"Inner dimensions must match for matrix multiplication: "
            f"A has shape {a.shape}, B has shape {b.shape}"
        )
        assert a.dtype == b.dtype, f"dtype mismatch: A has {a.dtype}, B has {b.dtype}"

        return GEMMSignature(
            a_shape=list(a.shape),
            b_shape=list(b.shape),
            dtype=a.dtype,
            **kwargs,
        )

    def get_func_name(self) -> str:
        """Format: gemm_{dtype}_{m}x{k}x{n}"""
        m, k = self.a_shape
        _, n = self.b_shape

        name_items = [
            "gemm",
            str(self.dtype).removeprefix("torch."),
            f"{m}x{k}x{n}",
        ]
        return "_".join(name_items)

    def get_output_size(self) -> int:
        """Returns the size of the output tensor in bytes."""
        return math.prod(self.output_shape) * self.dtype.itemsize

    def get_nn_module(self) -> torch.nn.Module:
        class Linear(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b)

        return Linear()

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

        return (get(self.a_shape), get(self.b_shape))
