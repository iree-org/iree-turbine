# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union

import torch

_T = TypeVar("_T", bound="ModeBase")


class ModeBase:
    @classmethod
    def parse(cls: type[_T], spec: Union[str, None, _T]) -> _T:
        if spec is None:
            return cls.FORWARD
        if isinstance(spec, cls):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in cls.__members__:
            raise ValueError(
                f"For mode= argument, expected one of: "
                f"{', '.join(cls.__members__.keys())}"
            )
        return cls[spec]


class OpSignature(ABC):
    """Base class for operation signatures providing sufficient information for launching."""

    @property
    def force_single_dispatch(self) -> bool:
        """
        Whether to compile as a single dispatch for the boo driver.
        Default `False` can be overridden for specific ops.
        """
        return False

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

    @property
    @abstractmethod
    def func_name(self) -> str:
        """MLIR function name to use for the operation, unique across operations."""
        ...

    @abstractmethod
    def get_nn_module(self, **kwargs) -> torch.nn.Module:
        """Generates a PyTorch neural network module containing this operation."""
        ...

    @abstractmethod
    def get_output_size(self) -> int:
        """Returns the size of this operation outputs in bytes."""
        ...

    @property
    @abstractmethod
    def is_forward(self) -> bool:
        """Indicates whether this signature corresponds to the forward-mode
        kernel."""
        ...

    @abstractmethod
    def as_init_kwargs(self) -> dict[str, Any]:
        """Return a dictionary that can be used as kwargs of the __init__
        function to construct a copy of this signature."""
        ...

    @abstractmethod
    def arrange_backward_launch_args(
        self,
        forward_args: tuple[torch.Tensor, ...],
        forward_results: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor]:
        """Arranges arguments and results of the forward pass in a way that they
        can be passed as trailing arguments to the backward pass.

        In the backward pass, we make an assumption that the result derivative
        is always the leading argument. We may also need to "save" values from
        the forward pass for the backward pass. We assume it is done by
        returning these values from the forward pass.
        """
        ...

    @property
    def main_result_index(self) -> int:
        """The index of the 'main' result of the operation.

        All other results are expected to be used as a mechanism to pass values
        between the forward and the backward pass. Therefore, various mechanisms
        will only consider propagating derivatives through this result. The
        design of at least the numerics testing must be revised if we ever need
        truly multi-result operations.

        Derived classes may override this.
        """
        return 0
