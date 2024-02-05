# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python API for IREE's high-level tensor dialects."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import functools

import torch

from ....support.ir_imports import (
    IndexType,
    IntegerType,
    IrType,
    RankedTensorType,
    StringAttr,
    Value,
    arith_d,
    flow_d,
)

from ....support.conversions import (
    TORCH_DTYPE_TO_IREE_TYPE,
)

from ..ir_utils import (
    build_index_value,
)

from .base import (
    Intrinsic,
    current_ir_trace,
    ShapedTypeDynamicSizeSentinel,
)

from .primitives import (
    IrScalar,
    IrImmediateScalar,
    IrTensor,
    IrImmediateTensor,
)

BuildableScalarValue = Union[IrScalar, Value]
BuildableTensorDimDecl = Union[int, Value]
BuildableTensorType = IrTensor
BuildableIndexType = Union[BuildableScalarValue, int]
BuildableIndexLengthType = Union[
    BuildableTensorDimDecl, Tuple[BuildableTensorDimDecl, BuildableTensorDimDecl]
]
BuildableSliceType = Sequence[BuildableIndexLengthType]
StaticIndexType = int


def cast_scalar_value(x: BuildableScalarValue) -> Value:
    x = unwrap_intrinsic_value(x)
    if not isinstance(x, Value):
        raise ValueError(f"Expected a scalar value but got {x}")
    return x


def cast_tensor_value(x: BuildableTensorType) -> IrTensor:
    assert isinstance(x, IrTensor), f"Expected a tensor but got {type(x)}"
    return x


def cast_index_value(
    x: BuildableIndexType, *, constant_cache: Optional[Dict[int, Value]] = None
) -> Value:
    x = unwrap_intrinsic_value(x)
    if isinstance(x, int):
        return build_index_value(x, constant_cache=constant_cache)
    else:
        return x


def cast_static_bounded_index(x: int, min_value: int, max_value: int) -> int:
    if not isinstance(x, int):
        raise ValueError(f"Expected int but got {type(x)}")
    if x < min_value or x > max_value:
        raise ValueError(
            f"Expected int in range [{min_value}, {max_value}] but got {x}"
        )
    return x


def cast_tensor_dim_decl(
    xs: Sequence[BuildableTensorDimDecl],
) -> Tuple[Sequence[int], Sequence[Value]]:
    """Casts a sequence of tensor declaration dimensions to dims suitable
    for construction of a TensorType and a sequence of dynamic dim values."""
    dim_decls: List[int] = []
    dynamic_dim_values: List[Value] = []
    for x in xs:
        x = unwrap_intrinsic_value(x)
        if isinstance(x, Value):
            assert_value_is_index(x)
            dim_decls.append(ShapedTypeDynamicSizeSentinel)
            dynamic_dim_values.append(x)
        elif isinstance(x, int) and x >= 0:
            dim_decls.append(x)
        else:
            raise ValueError(
                f"Expected a tensor dimension as a positive integer or None but got {x}"
            )
    return dim_decls, dynamic_dim_values


def cast_scalar_to_element_type(scalar: Value, element_type: IrType) -> Value:
    scalar_type = scalar.type
    # Support cast from Index -> Integer.
    if scalar_type == IndexType.get() and IntegerType.isinstance(element_type):
        return arith_d.IndexCastUIOp(element_type, scalar).result
    raise ValueError(
        f"Provided splat value ({type(value)}) does not match dtype {dtype} (and cannot be cast)"
    )


def assert_value_is_index(x: Value):
    t = x.type
    if not IndexType.isinstance(t):
        raise ValueError(f"Expected an index value but got {t}")


def unwrap_intrinsic_value(x) -> Any:
    if isinstance(x, Intrinsic):
        x, *rest = x.resolve_ir_values(current_ir_trace())
        if rest:
            raise ValueError(
                f"Expected a value that has an arity of one component but for {len(rest) + 1}"
            )
    return x


def emitter(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t = current_ir_trace()
        with t.loc, t.ip:
            return f(*args, **kwargs)

    return wrapper


class IREEEmitter:
    @emitter
    def tensor_dim(
        self,
        source: BuildableTensorType,
        index: int,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> "IrScalar":
        """Gets the dimension size of a tensor at a static position."""
        source = cast_tensor_value(source)
        index = cast_static_bounded_index(index, 0, source.rank - 1)
        dim_value = source.get_dim_value(index)
        if dtype is not None:
            try:
                cast_type = TORCH_DTYPE_TO_IREE_TYPE[dtype]()
            except KeyError:
                raise ValueError(f"Could not map Torch dtype {dtype} to an IREE type")
            dim_value = arith_d.IndexCastUIOp(cast_type, dim_value).result
        return IrImmediateScalar(dim_value)

    @emitter
    def tensor_empty(
        self, *dims: BuildableTensorDimDecl, dtype: torch.dtype = torch.float32
    ) -> IrTensor:
        """Constructs a tensor with uninitialized values.

        TODO: Support an IREE/raw element type in addition to the torch dtype.
        See: https://github.com/nod-ai/SHARK-Turbine/issues/130
        """
        dim_decls, dyn_dim_values = cast_tensor_dim_decl(dims)
        try:
            element_type = TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise ValueError(f"Could not map Torch dtype {dtype} to an IREE type")
        tensor_type = RankedTensorType.get(dim_decls, element_type)
        raw_tensor = flow_d.TensorEmptyOp(tensor_type, dyn_dim_values).result
        result = IrImmediateTensor(raw_tensor, dtype=dtype)
        result.set_dynamic_dim_values(dyn_dim_values)
        return result

    @emitter
    def tensor_reshape(
        self, source: BuildableTensorType, *result_dims: BuildableTensorDimDecl
    ) -> "IrTensor":
        constant_cache: Dict[int, Value] = {}
        source = cast_tensor_value(source)
        result_dim_decls, result_dynamic_dims = cast_tensor_dim_decl(result_dims)
        result_type = RankedTensorType.get(
            result_dim_decls, source.ir_type.element_type
        )
        result_value = flow_d.TensorReshapeOp(
            result_type,
            source.ir_value,
            source.get_only_dynamic_dim_values(constant_cache=constant_cache),
            result_dynamic_dims,
        ).result
        result = IrImmediateTensor(result_value, dtype=source.dtype)
        result.set_dynamic_dim_values(result_dynamic_dims)
        return result

    @emitter
    def tensor_slice(
        self, source: BuildableTensorType, *indices: BuildableSliceType
    ) -> "IrTensor":
        """Extracts a slice of a tensor.

        The given indices must match the rank of the source and each index is
        interpreted as `(start_index[, length])`, where the `length` is taken
        to be 1 if only a single value is given for an index.
        """
        source = cast_tensor_value(source)
        source_value = source.ir_value
        rank = source.rank
        if len(indices) != rank:
            raise ValueError(
                f"Slice indices must match the source rank. Got {len(indices)}, expected {rank}"
            )
        # Unpack start_indices and lengths.
        start_indices: List[BuildableIndexType] = []
        lengths: List[BuildableIndexType] = []
        for index_pack in indices:
            if isinstance(index_pack, (tuple, list)):
                if len(index_pack) == 2:
                    start_indices.append(index_pack[0])
                    lengths.append(index_pack[1])
                    continue
            else:
                start_indices.append(index_pack)
                lengths.append(1)
                continue
            raise ValueError(
                f"Slice indices expected to be a single value or a 2-tuple. Got {index_pack}"
            )

        # Process the lengths into a result shape and input length.
        index_value_cache: Dict[int, Value] = {}
        length_values: List[Value] = []
        result_shape: List[int] = []
        result_dynamic_dims: List[Value] = []
        for raw_length in lengths:
            if isinstance(raw_length, int):
                # Static.
                result_shape.append(raw_length)
                if raw_length in index_value_cache:
                    # Cached.
                    length_values.append(index_value_cache[raw_length])
                else:
                    # Not cached.
                    length_value = cast_index_value(raw_length)
                    index_value_cache[raw_length] = length_value
                    length_values.append(length_value)
            else:
                # Dynamic.
                result_shape.append(ShapedTypeDynamicSizeSentinel)
                length_value = cast_index_value(raw_length)
                length_values.append(length_value)
                result_dynamic_dims.append(length_value)
        assert len(length_values) == rank
        assert result_shape.count(ShapedTypeDynamicSizeSentinel) == len(
            result_dynamic_dims
        )

        # Process start indices.
        start_index_values = [cast_index_value(idx) for idx in start_indices]
        # Emit.
        result_type = RankedTensorType.get(result_shape, source.ir_type.element_type)
        constant_cache: Dict[int, Value] = {}
        result_value = flow_d.TensorSliceOp(
            result_type,
            source_value,
            source.get_only_dynamic_dim_values(constant_cache=constant_cache),
            start_index_values,
            length_values,
            result_dynamic_dims,
        ).result
        result = IrImmediateTensor(result_value, dtype=source.dtype)
        result.set_dynamic_dim_values(result_dynamic_dims)
        return result

    @emitter
    def tensor_update(
        self,
        target: BuildableTensorType,
        update: BuildableTensorType,
        *start_indices: BuildableIndexType,
    ) -> "IrTensor":
        """Applies an update to a target at start_indices and returns the mutated target."""
        constant_cache: Dict[int, Value] = {}
        target = cast_tensor_value(target)
        target_dynamic_dims = target.get_only_dynamic_dim_values(
            constant_cache=constant_cache
        )
        update = cast_tensor_value(update)
        update_dynamic_dims = update.get_only_dynamic_dim_values(
            constant_cache=constant_cache
        )
        start_index_dim_values = [
            cast_index_value(idx, constant_cache=constant_cache)
            for idx in start_indices
        ]
        result_value = flow_d.TensorUpdateOp(
            target.ir_value,
            target_dynamic_dims,
            start_index_dim_values,
            update.ir_value,
            update_dynamic_dims,
        ).result
        result = IrImmediateTensor(result_value, target.dtype)
        result.set_dynamic_dim_values(target_dynamic_dims)
        return result

    @emitter
    def tensor_splat(
        self,
        *dims: BuildableTensorDimDecl,
        value: BuildableScalarValue,
        dtype: torch.dtype,
    ) -> "IrTensor":
        # TODO: Type infer the dtype if missing.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/125
        dim_decls, dyn_dim_values = cast_tensor_dim_decl(dims)
        try:
            element_type = TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise ValueError(f"Could not map Torch dtype {dtype} to an IREE type")
        value = cast_scalar_value(value)
        if value.type != element_type:
            value = cast_scalar_to_element_type(value, element_type)
        tensor_type = RankedTensorType.get(dim_decls, element_type)
        raw_tensor = flow_d.TensorSplatOp(tensor_type, value, dyn_dim_values).result
        result = IrImmediateTensor(raw_tensor, dtype=dtype)
        result.set_dynamic_dim_values(dyn_dim_values)
        return result

    @emitter
    def tensor_trace(self, key: str, *ts: BuildableTensorType):
        dynamic_dims = []
        for t in ts:
            dynamic_dims.extend(t.get_only_dynamic_dim_values())
        ts = [cast_tensor_value(t).ir_value for t in ts]
        flow_d.TensorTraceOp(StringAttr.get(key), ts, dynamic_dims)


# Circular imports to resolve typing.
from .primitives import (
    IrScalar,
    IrTensor,
)
