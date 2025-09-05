# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from typing import Sequence, Collection, TypeVar

_T = TypeVar("_T")


class Permutation:
    """Composable and invertible lists which represent the second argument of `torch.permute`."""

    def __init__(self, ordering: Sequence[int]):
        assert list(sorted(ordering)) == list(
            range(len(ordering))
        ), "ordering must be rearragement of [0,1,2,...,n-1]"
        self._items = tuple(ordering)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[int, ...]:
        return self._items

    def __getitem__(self, n: int) -> int:
        return self.items[n]

    def __repr__(self) -> str:
        return f"Permutation of {self.size} : {self.items}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return False
        return self.items == other.items

    def __len__(self) -> int:
        return self.size

    def __mul__(self, other: "Permutation") -> "Permutation":
        """mimics composition `torch.permute(torch.permute(a, p1), p0) = torch.permute(a, p0*p1)"""
        assert self.size == other.size, "permutations must be the same size"
        return Permutation([other.items[element] for element in self.items])

    def __call__(self, other: torch.Tensor | Collection[_T]) -> torch.Tensor | list[_T]:
        """apply the permutation to a tensor or iterable (e.g., a shape)"""
        if isinstance(other, torch.Tensor):
            assert (
                len(other.shape) == self.size
            ), f"permutation must match the rank of the tensor being permuted, got permutation size {self.size} for tensor of shape {other.shape}"
            return torch.permute(other, self.items)
        if isinstance(other, Collection):
            assert len(other) == self.size
            return [other[item] for item in self.items]
        raise TypeError(f"Unexpected argument type: {type(other)}.")

    def __truediv__(self, other: "Permutation") -> "Permutation":
        return self * other.inv()

    def inv(self) -> "Permutation":
        """inverts the permutation x*inv(x) = inv(x)*x = Permutation.identity(x.size)"""
        inverse = list(range(self.size))
        for i in range(self.size):
            index = self.items[i]
            inverse[index] = i
        return Permutation(inverse)

    @staticmethod
    def identity(size: int) -> "Permutation":
        """creates an identity permutation"""
        assert size > 0, "size must be positive integer"
        return Permutation(list(range(size)))

    @staticmethod
    def get(src: Collection[_T], target: Collection[_T]) -> "Permutation":
        """Gets a permutation p such that `torch.permute(a, p) = b` where `a.shape = src` and `b.shape = target`"""
        n = len(src)
        assert n > 0 and n == len(
            target
        ), "source and target iterables must share the same positive length"
        d = {t: i for i, t in enumerate(target)}
        inverse = []
        try:
            for item in src:
                value = d.pop(item)
                inverse.append(value)
        except KeyError as e:
            raise ValueError(
                f"src and target should be permutations of a common set of unique items, got {src=}, {target=}"
            )
        return Permutation(inverse).inv()


def permute_layout(
    tensor: torch.Tensor, permutation: Permutation | Sequence[int]
) -> torch.Tensor:
    """Returns a new tensor that is the given permutation of the input tensor.

    The resulting tensor is stored in the contiguous format after permutation
    and its shape/strides are adjusted to match the shape of the original
    tensor.
    """
    if not isinstance(permutation, Permutation):
        permutation = Permutation(permutation)
    permuted = permutation(tensor)
    rematerialized = permuted.clone(memory_format=torch.contiguous_format)
    return permutation.inv()(rematerialized)
