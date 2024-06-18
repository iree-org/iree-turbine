from typing import Optional, Type, TypeVar, ClassVar

from .kernel_buffer import AddressSpace, KernelBufferMeta, KernelBufferUsage
from ..ops.wave_ops import register
from .._support.dtype import DataType
from .._support.indexing import IndexExpr

__all__ = [
    "Memory",
    "Register",
]

MemoryTypeT = TypeVar("MemoryTypeT")


class Memory(metaclass=KernelBufferMeta):
    """
    Represents storage anywhere in the memory hierarchy except registers.
    Parameterized by a shape, address space and element type. The allocated
    memory is traversed by an iterator that specifies the offset, stride
    and size along each dimension.
    """

    address_space: ClassVar[int]
    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    usage: ClassVar[Optional[KernelBufferUsage]]

    def __init__(self) -> None:
        raise NotImplementedError("Memory types are not directly instantiated.")

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Memory"]:
        """Syntax: `Memory[shape1, ...., shapeN, addressSpace, dtype, Optional[usage]]"""
        if len(shape_and_dtype) < 3:
            raise TypeError(f"Expected at least 3 arguments, got: {shape_and_dtype}")

        shift = 0
        usage = KernelBufferUsage.NONE
        if isinstance(shape_and_dtype[-1], KernelBufferUsage):
            shift = 1
            usage = shape_and_dtype[-1]
        shape = shape_and_dtype[: -2 - shift]
        addressSpace = shape_and_dtype[-2 - shift]
        dtype = shape_and_dtype[-1 - shift]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)
        if not all(isinstance(s, IndexExpr) for s in shape) or len(shape) == 0:
            raise TypeError(f"Expected shape to be a tuple of IndexExpr, got {shape}")
        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")
        if not (
            isinstance(addressSpace, IndexExpr)
            or isinstance(addressSpace, AddressSpace)
        ):
            raise TypeError(
                f"Expected addressSpace to be a AddressSpace, got {addressSpace}"
            )
        if addressSpace == AddressSpace.REGISTER:
            raise TypeError(
                f"Memory does not support address space register, use Register instead."
            )

        return cls.new_subtype(
            name="Memory",
            address_space=addressSpace,
            symbolic_shape=shape,
            dtype=dtype,
            usage=usage,
        )


class Register(metaclass=KernelBufferMeta):
    """
    Represents virtual registers. Parameterized by a shape and element type.
    Instantiating this class emits a new `register` operation.
    """

    symbolic_shape: ClassVar[tuple[IndexExpr, ...]]
    rank: ClassVar[int]
    dtype: ClassVar[DataType]
    value: float

    def __new__(cls, value: float) -> "Register":
        return register(cls.symbolic_shape, cls.dtype, value)

    def __class_getitem__(
        cls, shape_and_dtype: tuple[IndexExpr | DataType, ...]
    ) -> Type["Register"]:

        if len(shape_and_dtype) < 2:
            raise TypeError(f"Expected at least 2 arguments, got: {shape_and_dtype}")

        shape = shape_and_dtype[:-1]
        dtype = shape_and_dtype[-1]

        # Allow constant int expressions in shape
        shape = tuple(IndexExpr(s) if isinstance(s, int) else s for s in shape)

        if not isinstance(dtype, DataType):
            raise TypeError(f"Expected dtype to be a DataType, got {dtype}")

        return cls.new_subtype(
            name="Register",
            address_space=AddressSpace.REGISTER,
            symbolic_shape=shape,
            dtype=dtype,
        )
