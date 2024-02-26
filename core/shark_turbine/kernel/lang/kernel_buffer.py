from typing import Type, TypeVar, cast, ClassVar

from enum import Enum

import torch

from .._support.indexing import IndexExpr
from .._support.shaped_type import ShapedDataType
from .._support.dtype import DataType, f32
from .. import ops

__all__ = [
    "KernelBuffer",
    "InputBuffer",
    "OutputBuffer",
    "TemporaryBuffer",
    "is_kernel_buffer_meta_derived",
]

SubtypeT = TypeVar("SubtypeT")


class NotSetType:
    ...


NotSet = NotSetType()


class KernelBufferUsage(Enum):
    NONE = 0
    INPUT = 1
    OUTPUT = 2
    TEMPORARY = 3

    @staticmethod
    def _type_name(v) -> str:
        if v == KernelBufferUsage.NONE:
            return "KernelBuffer"
        elif v == KernelBufferUsage.INPUT:
            return "InputBuffer"
        elif v == KernelBufferUsage.OUTPUT:
            return "OutputBuffer"
        elif v == KernelBufferUsage.TEMPORARY:
            return "TemporaryBuffer"
        else:
            raise AssertionError(f"uncovered KernelBufferUsage enum ({v})")


class _KernelBufferMeta(ShapedDataType):
    usage: KernelBufferUsage = KernelBufferUsage.NONE

    def new_subtype(
        cls: Type[SubtypeT],
        *,
        symbolic_shape: tuple[IndexExpr, ...] | NotSetType = NotSet,
        dtype: DataType | NotSetType = NotSet,
        usage: KernelBufferUsage | NotSetType = NotSet,
    ) -> Type[SubtypeT]:
        init_symbolic_shape = symbolic_shape if symbolic_shape is not NotSet else cls.symbolic_shape  # type: ignore
        init_dtype = dtype if dtype is not NotSet else cls.dtype  # type: ignore
        init_usage = usage if usage is not NotSet else cls.usage  # type: ignore

        class SubType(cls):
            symbolic_shape = init_symbolic_shape
            rank = len(init_symbolic_shape)  # type: ignore
            dtype = init_dtype
            usage = init_usage

        SubType.__name__ = KernelBufferUsage._type_name(init_usage)

        return cast(Type[SubtypeT], SubType)


class KernelBuffer(metaclass=_KernelBufferMeta):
    """Represents a buffer in global memory.

    Top level kernels always operate on global memory via these
    buffers, and the primary operations that can be performed on
    them are loads/stores and DMAs to some form of compute
    capable local buffer.

    When executing eagerly, these are backed by a normal torch
    Tensor. When compiling, an appropriate duck-typed proxy
    is used.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]

    def __init__(self, tensor: torch.Tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected Tensor but got {tensor}"
        type_rank = type(self).rank
        tensor_rank = len(tensor.shape)
        if type_rank is not None and type_rank != tensor_rank:
            raise ValueError(
                f"Cannot create {type(self)}(tensor({tensor.shape})): mismatched symbolic rank"
            )
        self._tensor = tensor

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["KernelBuffer"]:
        """Syntax: `KernelBuffer[shape1, shape2, ..., shapeN, dtype]`"""

        if not isinstance(shape_and_dtype, tuple) or len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        if not all(isinstance(s, IndexExpr) for s in shape):
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")

        shape = cast(tuple[IndexExpr, ...], shape)
        dtype = cast(DataType, dtype)

        return cls.new_subtype(symbolic_shape=shape, dtype=dtype)

    def __repr__(self):
        return f"{type(self)}({self._tensor})"

    def __setitem__(self, key, item):
        ops.kernel_buffer_setitem(self, key, item)

    def __getitem__(self, key):
        return ops.kernel_buffer_getitem(self, key)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tensor.shape


class InputBuffer(KernelBuffer):
    usage = KernelBufferUsage.INPUT


class OutputBuffer(KernelBuffer):
    usage = KernelBufferUsage.OUTPUT


class TemporaryBuffer(KernelBuffer):
    usage = KernelBufferUsage.TEMPORARY


def is_kernel_buffer_meta_derived(t: type) -> bool:
    return isinstance(t, _KernelBufferMeta)
