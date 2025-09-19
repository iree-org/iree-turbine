# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from dataclasses import dataclass

import torch


__all__ = [
    "DeviceAffinity",
    "DeviceTensorTrait",
    "ExternalTensorTrait",
]


@dataclass(frozen=True)
class DeviceAffinity:
    """This is used to provide device affinities to exported function arguments."""

    ordinal: int
    queues: Optional[list] = None

    def __repr__(self) -> str:
        if self.queues is None:
            return f"DeviceAffinity({self.ordinal})"
        return f"DeviceAffinity({self.ordinal}, [{', '.join(self.queues)}])"


@dataclass
class DeviceTensorTrait:
    """Represents a 'trait' that can be applied to a Tensor to signal that
    it is to be loaded to a speific device at execution time.
    """

    ordinal: int
    queues: Optional[list] = None

    @staticmethod
    def get(from_tensor: torch.Tensor) -> Optional["DeviceTensorTrait"]:
        existing = getattr(from_tensor, "_turbine_device_tensor_trait", None)
        if existing is None:
            return None
        assert isinstance(existing, DeviceTensorTrait)
        return existing

    def set(self, to_tensor: torch.Tensor):
        to_tensor._turbine_device_tensor_trait = self  # type: ignore


@dataclass
class ExternalTensorTrait:
    """Represents a 'trait' that can be applied to a Tensor to signal that
    it is to be loaded by name from an external archive at AOT execution time.
    """

    external_scope: str
    external_name: str

    @staticmethod
    def get(from_tensor: torch.Tensor) -> Optional["ExternalTensorTrait"]:
        existing = getattr(from_tensor, "_turbine_external_tensor_trait", None)
        if existing is None:
            return None
        assert isinstance(existing, ExternalTensorTrait)
        return existing

    def set(self, to_tensor: torch.Tensor):
        to_tensor._turbine_external_tensor_trait = self  # type: ignore
