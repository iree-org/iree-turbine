from typing import Type, TypeVar, cast, ClassVar

from enum import Enum
from dataclasses import dataclass

import torch

from .._support.indexing import IndexExpr
from .._support.shaped_type import ShapedDataType
from .._support.dtype import DataType, f32
from .. import ops

__all__ = [
    "AddressSpace",
    "KernelBuffer",
    "InputBuffer",
    "MemoryLayout",
    "OutputBuffer",
    "TemporaryBuffer",
    "is_kernel_buffer_meta_derived",
]

SubtypeT = TypeVar("SubtypeT")


class AddressSpace(Enum):
    REGISTER = 0
    SHARED_MEMORY = 1
    GLOBAL_MEMORY = 2


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


@dataclass
class MemoryLayout:
    """
    Specifies the physical layout of a memory buffer in terms of
    its physical shape.
    """

    shape: tuple[int | IndexExpr]


class KernelBufferMeta(ShapedDataType):
    usage: KernelBufferUsage = KernelBufferUsage.NONE

    def new_subtype(
        cls: Type[SubtypeT],
        *,
        name: str | None = None,
        address_space: AddressSpace | None = None,
        symbolic_shape: tuple[IndexExpr, ...] | None = None,
        dtype: DataType | None = None,
        physical_layout: MemoryLayout | None = None,
        usage: KernelBufferUsage | None = None,
    ) -> Type[SubtypeT]:
        init_address_space = (
            address_space if address_space else AddressSpace.GLOBAL_MEMORY
        )
        init_symbolic_shape = symbolic_shape if symbolic_shape is not None else cls.symbolic_shape  # type: ignore
        init_dtype = dtype if dtype is not None else cls.dtype  # type: ignore
        init_physical_layout = physical_layout if physical_layout else None  # type: ignore
        init_usage = usage if usage is not None else cls.usage  # type: ignore

        class SubType(cls):
            address_space = init_address_space
            symbolic_shape = init_symbolic_shape
            rank = len(init_symbolic_shape)  # type: ignore
            dtype = init_dtype
            physical_layout = init_physical_layout
            usage = init_usage

        if name is not None:
            SubType.__name__ = name
        else:
            SubType.__name__ = KernelBufferUsage._type_name(init_usage)

        return cast(Type[SubtypeT], SubType)


class KernelBuffer(metaclass=KernelBufferMeta):
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
    return isinstance(t, KernelBufferMeta)
