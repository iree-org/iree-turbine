# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Optional

import torch


class OpSignature(ABC):
    """Base class for operation signatures providing sufficient information for launching."""

    @abstractmethod
    def get_sample_args(
        self,
        *,
        device: str | torch.device | None = None,
        splat_value: int | float | None = None,
        seed: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """Generates sample arguments as PyTorch tensors for the operation."""
        ...

    @abstractmethod
    def get_func_name(self) -> str:
        """Generates an MLIR function name to use for the operation, unique across operations."""
        ...

    @abstractmethod
    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        """Generates a PyTorch neural network module containing this operation."""
        ...

    @abstractmethod
    def get_output_size(self) -> int:
        """Returns the size of this operation outputs in bytes."""
        ...
