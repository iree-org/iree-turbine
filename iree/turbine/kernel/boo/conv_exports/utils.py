# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Collection,
    List,
    Tuple,
)

import torch

__all__ = ["Permutation"]


class Permutation:
    """Composable and invertible lists which represent the second argument of `torch.permute`."""

    def __init__(self, ordering: List[int]):
        assert list(sorted(ordering)) == list(
            range(len(ordering))
        ), "ordering must be rearragement of [0,1,2,...,n-1]"
        self._items = tuple(ordering)

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def items(self) -> Tuple[int, ...]:
        return self._items

    def __getitem__(self, n: int):
        return self.items[n]

    def __repr__(self):
        return f"Permutation of {self.size} : {self.items}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return False
        return self.items == other.items

    def __len__(self):
        return self.size

    def __mul__(self, other: "Permutation"):
        """mimics composition `torch.permute(torch.permute(a, p1), p0) = torch.permute(a, p0*p1)"""
        assert self.size == other.size, "permutations must be the same size"
        return Permutation([other.items[element] for element in self.items])

    def __call__(self, other):
        """apply the permutation to a tensor or iterable (e.g., a shape)"""
        if isinstance(other, torch.Tensor):
            assert (
                len(other.shape) == self.size
            ), f"permutation must match the rank of the tensor being permuted, got permutation size {self.size} for tensor of shape {other.shape}"
            return torch.permute(other, self.items)
        if isinstance(other, Collection):
            assert len(other) == self.size
            return [other[item] for item in self.items]

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
    def identity(size: int):
        """creates an identity permutation"""
        assert size > 0, "size must be positive integer"
        return Permutation(list(range(size)))

    @staticmethod
    def get(src: Collection, target: Collection):
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
