# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Iterator, List, Optional, Set, Tuple, Union

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from iree.runtime import (
    ParameterIndex,
    ParameterIndexEntry,
)

from .tensor_traits import (
    ExternalTensorTrait,
)


__all__ = [
    "externalize_module_parameters",
    "save_module_parameters",
    "ParameterArchive",
    "ParameterArchiveBuilder",
]

################################################################################
# Parameter externalization
################################################################################


def externalize_module_parameters(
    module: nn.Module, *, external_scope: str = "", prefix: str = ""
):
    """Externalizes parameters and persistent buffers in a module by name."""

    for tensor_name, tensor in _yield_saveable_tensors(module, prefix=prefix):
        trait = ExternalTensorTrait(
            external_scope=external_scope, external_name=tensor_name
        )
        trait.set(tensor)


################################################################################
# Parameter archives save/load
################################################################################


def save_module_parameters(
    file_path: Union[str, Path], module: nn.Module, *, prefix: str = ""
):
    """One shot save of parameters and persistent buffers on a module.

    More options are available by using a ParameterArchiveBuilder.
    """
    builder = ParameterArchiveBuilder()
    builder.add_module(module, prefix=prefix)
    builder.save(file_path)


class ParameterArchive:
    """Allows access to a parameter archive as CPU tensors.

    TODO: Add more helpers for reading tensors once we get upstream versions that
    have that integrated.
    """

    def __init__(
        self, file_path: Optional[Union[str, Path]] = None, *, mmap: bool = True
    ):
        self._index = ParameterIndex()
        if file_path is not None:
            self.load(file_path, mmap=mmap)

    def load(
        self,
        file_path: Union[str, Path],
        *,
        mmap: bool = True,
        readable: bool = True,
        writable: bool = False,
    ):
        """Loads index entries from a file adding them to the in-memory archive."""
        self._index.load(
            str(file_path), mmap=mmap, readable=readable, writable=writable
        )

    @property
    def index(self) -> ParameterIndex:
        return self._index

    def items(self) -> List[Tuple[str, ParameterIndexEntry]]:
        return self._index.items()

    def __repr__(self):
        return repr(self._index)


class ParameterArchiveBuilder:
    """Helper for building parameter archives from live modules."""

    def __init__(self):
        self._index = ParameterIndex()

    def save(self, file_path: Union[str, Path]):
        """Saves the archive."""
        self._index.create_archive_file(str(file_path))

    def add_tensor(self, name: str, tensor: torch.Tensor):
        """Adds an named tensor to the archive."""
        host_array = tensor.detach().cpu().contiguous().numpy()
        self._index.add_buffer(name, host_array)

    def add_module(self, module: nn.Module, *, prefix: str = ""):
        """Adds all parameters and persistent buffers from a module hierarchy."""
        for name, t in _yield_saveable_tensors(module, prefix=prefix):
            self.add_tensor(name, t)


def _yield_saveable_tensors(
    module: nn.Module, *, prefix: str = ""
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yields tuple of name/tensor for all saveable tensors in a module.

    This includes parameters and persistent buffers.
    """
    memo: Set[str] = set()
    for sub_name, sub_module in module.named_modules(prefix=prefix):
        state_dict = sub_module.state_dict()
        for param_name, param in sub_module.named_parameters(recurse=False):
            full_param_name = f"{sub_name}.{param_name}" if sub_name else param_name
            if full_param_name in memo:
                continue
            memo.add(full_param_name)
            yield full_param_name, param
        for buffer_name, buffer in sub_module.named_buffers(recurse=False):
            full_buffer_name = f"{sub_name}.{buffer_name}" if sub_name else buffer_name
            if full_buffer_name in memo:
                continue
            memo.add(full_buffer_name)
            if buffer_name not in state_dict:
                # Non persistent
                continue
            yield full_buffer_name, buffer
