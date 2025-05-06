# Copyright 2023 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable
from collections.abc import Sequence

import numpy as np
import torch

from iree.runtime import (
    HalElementType,
)

from iree.compiler.extras.fx_importer import (
    TORCH_DTYPE_TO_MLIR_TYPE_ASM,
)

from .exceptions import (
    UnknownDTypeError,
)

from .ir_imports import (
    Float8E4M3FNUZType,
    BF16Type,
    ComplexType,
    F16Type,
    F32Type,
    F64Type,
    IntegerType,
    IrType,
    Float8E4M3FNType,
    Float8E4M3FNUZType,
    Float8E5M2Type,
    Float8E5M2FNUZType,
)

# We need the inverse of the TORCH_DTYPE_TO_MLIR_TYPE_ASM table.
MLIR_TYPE_ASM_TO_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_TO_MLIR_TYPE_ASM.items()}

# When emitting constants, we have to create native IREE types.
TORCH_DTYPE_TO_IREE_TYPE: dict[torch.dtype, Callable[[], IrType]] = {
    torch.float8_e4m3fnuz: lambda: Float8E4M3FNUZType.get(),
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.uint8: lambda: IntegerType.get_signless(8),
    torch.int8: lambda: IntegerType.get_signless(8),
    torch.int16: lambda: IntegerType.get_signless(16),
    torch.int32: lambda: IntegerType.get_signless(32),
    torch.int64: lambda: IntegerType.get_signless(64),
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.qint8: lambda: IntegerType.get_signless(8),
    torch.quint8: lambda: IntegerType.get_signless(8),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
    torch.complex128: lambda: ComplexType.get(F64Type.get()),
    torch.float8_e4m3fn: lambda: Float8E4M3FNType.get(),
    torch.float8_e4m3fnuz: lambda: Float8E4M3FNUZType.get(),
    torch.float8_e5m2: lambda: Float8E5M2Type.get(),
    torch.float8_e5m2fnuz: lambda: Float8E5M2FNUZType.get(),
}

TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "ui8",
    torch.int8: "si8",
    torch.int16: "si16",
    torch.int32: "si32",
    torch.int64: "si64",
    torch.bool: "i1",
    torch.qint8: "si8",
    torch.quint8: "ui8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
    torch.float8_e4m3fn: "f8E4M3EFN",
    torch.float8_e4m3fnuz: "f8E4M3EFNUZ",
    torch.float8_e5m2: "f8E5M2E",
    torch.float8_e5m2fnuz: "f8E5M2EFNUZ",
}

SIGNED_MLIR_TYPE_ASM_TO_TORCH_DTYPE = dict(
    (v, k) for k, v in TORCH_DTYPE_TO_SIGNED_MLIR_TYPE_ASM.items()
)

TORCH_DTYPE_TO_IREE_TYPE_ASM = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
    torch.float64: "f64",
    torch.uint8: "i8",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "i1",
    torch.qint8: "i8",
    torch.quint8: "i8",
    torch.complex32: "complex<f16>",
    torch.complex64: "complex<f32>",
    torch.complex128: "complex<f64>",
    torch.float8_e4m3fn: "f8E4M3FN",
    torch.float8_e4m3fnuz: "f8E4M3FNUZ",
    torch.float8_e5m2: "f8E5M2",
    torch.float8_e5m2fnuz: "f8E5M2FNUZ",
}

IREE_TYPE_ASM_TO_TORCH_DTYPE = dict(
    (v, k) for k, v in TORCH_DTYPE_TO_IREE_TYPE_ASM.items()
)

DTYPE_TO_ELEMENT_TYPE: dict[torch.dtype, HalElementType] = {
    torch.float16: HalElementType.FLOAT_16,
    torch.bfloat16: HalElementType.BFLOAT_16,
    torch.float32: HalElementType.FLOAT_32,
    torch.float64: HalElementType.FLOAT_64,
    torch.uint8: HalElementType.UINT_8,
    torch.int8: HalElementType.SINT_8,
    torch.int16: HalElementType.SINT_16,
    torch.int32: HalElementType.SINT_32,
    torch.int64: HalElementType.SINT_64,
    torch.bool: HalElementType.BOOL_8,
    torch.qint8: HalElementType.OPAQUE_8,
    torch.quint8: HalElementType.OPAQUE_8,
    torch.complex64: HalElementType.COMPLEX_64,
    torch.complex128: HalElementType.COMPLEX_128,
    torch.float8_e4m3fn: HalElementType.FLOAT_8_E4M3_FN,
    torch.float8_e4m3fnuz: HalElementType.FLOAT_8_E4M3_FNUZ,
    torch.float8_e5m2: HalElementType.FLOAT_8_E5M2,
    torch.float8_e5m2fnuz: HalElementType.FLOAT_8_E5M2_FNUZ,
}


def dtype_to_element_type(dtype) -> HalElementType:
    try:
        return DTYPE_TO_ELEMENT_TYPE[dtype]
    except KeyError:
        raise UnknownDTypeError(dtype)


TORCH_DTYPE_TO_NUMPY = {
    torch.float16: np.dtype("f2"),
    torch.float32: np.dtype("f4"),
    torch.float64: np.dtype("f8"),
    torch.uint8: np.dtype("u1"),
    torch.int8: np.dtype("i1"),
    torch.int16: np.dtype("i2"),
    torch.int32: np.dtype("i4"),
    torch.int64: np.dtype("i8"),
    torch.bool: np.dtype("?"),
    torch.complex64: np.dtype("c8"),
    torch.complex128: np.dtype("c16"),
    torch.bfloat16: np.dtype("i2"),
    torch.float8_e4m3fn: np.dtype("i1"),
    torch.float8_e4m3fnuz: np.dtype("i1"),
    torch.float8_e5m2: np.dtype("i1"),
    torch.float8_e5m2fnuz: np.dtype("i1"),
}


def torch_dtype_to_numpy(torch_dtype: torch.dtype) -> Any:
    try:
        return TORCH_DTYPE_TO_NUMPY[torch_dtype]
    except KeyError:
        raise UnknownDTypeError(torch_dtype)


def torch_dtyped_shape_to_iree_format(
    shape_or_tensor: Sequence[int] | torch.Tensor,
    /,
    dtype: torch.dtype | None = None,
) -> str:
    """Example:
    shape = [1, 2, 3]
    dtype = torch.bfloat16
    Returns
    "1x2x3xbf16"
    """
    if isinstance(shape_or_tensor, torch.Tensor):
        dtype = dtype or shape_or_tensor.dtype
        return torch_dtyped_shape_to_iree_format(shape_or_tensor.shape, dtype)
    else:
        if dtype is None:
            raise ValueError(
                "dtype must be provided when shape is given instead of a tensor."
            )
    shape_str = "x".join([str(d) for d in shape_or_tensor])
    shape_dtype_delimiter = "x" if len(shape_or_tensor) > 0 else ""
    dtype_str = TORCH_DTYPE_TO_IREE_TYPE_ASM[dtype]
    return f"{shape_str}{shape_dtype_delimiter}{dtype_str}"
