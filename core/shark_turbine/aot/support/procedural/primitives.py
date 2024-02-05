# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Live types during runtime of a procedure trace. User code will
# operate on instances of these.

from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from torch.export import (
    Constraint,
    dynamic_dim,
)

from ....support.ir_imports import (
    F32Type,
    IrType,
    RankedTensorType,
    Value,
    arith_d,
)

from ..ir_utils import (
    build_tensor_dim_value,
    _is_float_type,
    _is_integer_like_type,
)

from ..utils import (
    Empty,
    EmptyType,
)

from .base import (
    Intrinsic,
    IrTrace,
    ShapedTypeDynamicSizeSentinel,
    current_ir_trace,
)

###############################################################################
# Tensors and scalars
###############################################################################


class IrScalar(Intrinsic):
    """An intrinsic that represents a scalar value.

    Subclasses are responsible for providing either value or load semantics.
    """

    __slots__ = [
        "ir_type",
    ]

    def __init__(self, ir_type: IrType):
        self.ir_type = ir_type

    def set(self, other):
        t = current_ir_trace()
        with t.ip, t.loc:
            # Type check and promotion.
            # TODO: Add more comprehensive type promotion hiearchy.
            lhs = self.ir_value
            rhs = None
            if isinstance(other, IrScalar):
                # Assumes when both are Value, they have same type.
                rhs = other.ir_value
            elif isinstance(other, (int, bool)) and _is_integer_like_type(self.ir_type):
                rhs = arith_d.ConstantOp(lhs.type, other).result
            elif isinstance(other, (float)) and _is_float_type(self.ir_type):
                rhs = arith_d.ConstantOp(lhs.type, other).result
            if rhs is None or lhs.type != rhs.type:
                raise ValueError(
                    f"Cannot handle src type of {self.ir_type} to dst python type of {type(other)}."
                )
            return IrImmediateScalar(rhs)

    def __add__(self, other):
        t = current_ir_trace()
        with t.ip, t.loc:
            # Type check and promotion.
            # TODO: Add more comprehensive type promotion hiearchy as seen in
            # https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
            # See: https://github.com/nod-ai/SHARK-Turbine/issues/132
            lhs = self.ir_value
            if isinstance(other, IrScalar):
                # Assumes when both are Value, they have same type.
                rhs = other.ir_value
            elif isinstance(other, (int, bool)):
                rhs = arith_d.ConstantOp(lhs.type, other).result
            elif isinstance(other, float) and _is_integer_like_type(self.ir_type):
                lhs = arith_d.SIToFPOp(F32Type.get(), lhs).result
                rhs = arith_d.ConstantOp(F32Type.get(), other).result

            #  Checks that lhs and rhs has same type.
            if lhs.type != rhs.type:
                raise ValueError("Mismatch type between lhs and rhs.")

            # Emit computation.
            if _is_integer_like_type(lhs.type):
                return IrImmediateScalar(arith_d.AddIOp(lhs, rhs).result)
            elif _is_float_type(lhs.type):
                return IrImmediateScalar(arith_d.AddFOp(lhs, rhs).result)
            else:
                raise ValueError(
                    f"Expected operand to be either Int or Float but got {self.ir_type} instead."
                )


class IrImmediateScalar(IrScalar):
    """Represents an IR scalar value."""

    __slots__ = [
        "ir_value",
    ]

    def __init__(self, ir_value: Value):
        super().__init__(ir_value.type)
        assert isinstance(ir_value, Value)
        self.ir_value = ir_value

    def resolve_ir_values(self, proc_trace: IrTrace) -> Sequence[Value]:
        return (self.ir_value,)


class IrTensor(Intrinsic):
    """An intrinsic that represents a tensor value.

    Carries additional metadata needed to resolve dimensions and original
    PyTorch attributes.
    """

    __slots__ = [
        "ir_type",
        "dtype",
        "_cached_dim_values",
        "_dynamic_dims",
        "_shape",
        "_meta_tensor",
        "_meta_tensor_constraints",
    ]

    def __init__(self, ir_type: IrType, dtype: torch.dtype):
        assert isinstance(dtype, torch.dtype)
        ranked_ir_type = RankedTensorType(ir_type)
        self.ir_type = ranked_ir_type
        self.dtype = dtype
        # We always cache the meta tensor once asked for since it is used
        # to anchor constraints. The constraints list is the same size as
        # the rank and has a non-None dynamic_dim constraint for each
        # dynamic dimension in the type.
        self._meta_tensor: Optional[torch.Tensor] = None
        self._meta_tensor_constraints: Optional[List[Constraint]] = None

        # Figure dynamic dims.
        # _dynamic_dims is either Empty if static, or Value/None if dynamic.
        self._shape = ranked_ir_type.shape
        self._dynamic_dims: List[Union[EmptyType, Value, None]] = [
            None if d == ShapedTypeDynamicSizeSentinel else Empty for d in self._shape
        ]

        # If we computed a dim, then stash it here for later use.
        self._cached_dim_values: List[Optional[Value]] = [None] * len(
            self._dynamic_dims
        )

    def dynamic_dim(self, i: int) -> Constraint:
        """Access the dynamic_dim constraint for the i'th dimension."""
        self._populate_meta_tensor()
        c = self._meta_tensor_constraints[i]
        if c is None:
            raise TypeError(
                f"Requested dynamic_dim constraint for dimension {i} of {self.ir_type} which is not dynamic"
            )
        return c

    @property
    def rank(self) -> int:
        return len(self._shape)

    @property
    def dynamic_dim_count(self) -> int:
        return len(self._dynamic_dims) - self._dynamic_dims.count(Empty)

    def set_dim_value(self, index: int, value: Optional[Value]):
        """Sets the value of a dynamic dim.

        Raises ValueError if the dimension is not dynamic.
        """
        if self._dynamic_dims is Empty:
            raise ValueError(f"Dimension {index} of {self} is not dynamic")
        self._dynamic_dims[index] = value

    def set_dynamic_dim_values(self, values: Sequence[Value]):
        """Sets all dynamic dim values."""
        dd = self._dynamic_dims
        input_index = 0
        for pos in range(len(dd)):
            if dd[pos] is Empty:
                # Static
                continue
            assert input_index < len(values), "Mismatched static/dynamic dims"
            assert isinstance(values[input_index], Value)
            dd[pos] = values[input_index]
            input_index += 1
        assert input_index == len(values), "Mismatched static/dynamic dims"

    def get_dim_value(
        self,
        index: int,
        *,
        constant_cache: Optional[Dict[int, Value]] = None,
        resolved_ir_value: Optional[Value] = None,
    ) -> Value:
        """Gets a dimension as an Index value.

        Requires that an InsertionPoint and Location are on the context stack.

        This will cache the dim value, returning the cached value later if
        requested.
        """
        cached_dim = self._cached_dim_values[index]
        if cached_dim:
            return cached_dim
        dynamic_dim = self._dynamic_dims[index]
        if dynamic_dim is Empty or dynamic_dim is None:
            if resolved_ir_value is None:
                resolved_ir_value = self.ir_value
            # Construct a static dimension.
            # TODO: Add MLIR API support for creating an insertion point after
            # an operation and use that to set the InsertionPoint to the
            # earliest point.
            # See: https://github.com/nod-ai/SHARK-Turbine/issues/133
            dim_value = build_tensor_dim_value(
                resolved_ir_value, index, constant_cache=constant_cache
            )
            self._cached_dim_values[index] = dim_value
            return dim_value
        else:
            # Dynamic dim is known.
            return dynamic_dim

    def get_only_dynamic_dim_values(
        self,
        *,
        constant_cache: Optional[Dict[int, Value]] = None,
        resolved_ir_value: Optional[Value] = None,
    ) -> List[Value]:
        """Returns a list of *only* the dynamic dim Values."""
        values: List[Value] = []
        for i, sentinel in enumerate(self._dynamic_dims):
            if sentinel is not Empty:
                # Cache IR value so we don't materialize for each
                # dynamic dim.
                if resolved_ir_value is None:
                    resolved_ir_value = self.ir_value
                values.append(
                    self.get_dim_value(
                        i,
                        constant_cache=constant_cache,
                        resolved_ir_value=resolved_ir_value,
                    )
                )
        return values

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented

    def _populate_meta_tensor(self):
        if self._meta_tensor is not None:
            return

        ir_tensor_type = self.ir_type
        shape = ir_tensor_type.shape
        # TODO: We shouldn't need to create a real tensor here, as Dynamo will
        # immediately convert it to fake. However, it will also set up the shape
        # environment and asserts that any fake tensor inputs are from its
        # internal FakeMode. There should be a way but needs more investigation.
        # TODO: This tensor needs a device that matches the model being exported.
        # We just create these on the CPU because that is common.
        # Note that in Dynamo's modeling of dynamic shapes, 0/1 are specialized and
        # cannot be dynamic, and we must use a >= 2 dimension value to represent
        # a dynamic quantity. We therefore adjust the shape in this way and
        # add a dynamic_dim constraint.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/134
        extents = [2 if d < 0 else d for d in shape]
        mt = self._meta_tensor = torch.empty(extents, dtype=self.dtype)
        # Generate constraints that are aligned with any dynamic dimensions or None
        # if static.
        self._meta_tensor_constraints = [
            dynamic_dim(mt, i) if d < 0 else None for i, d in enumerate(shape)
        ]

    def _to_meta_tensor(self) -> Tuple[torch.Tensor, List[Constraint]]:
        """Converts to a fake Tensor that dynamo can handle."""
        self._populate_meta_tensor()
        return self._meta_tensor, [
            c for c in self._meta_tensor_constraints if c is not None
        ]


class IrImmediateTensor(IrTensor):
    """Represents a Value in the IR under construction during procedural tracing."""

    __slots__ = [
        "ir_value",
    ]

    def __init__(self, ir_value: Value, dtype: torch.dtype):
        super().__init__(ir_value.type, dtype)
        self.ir_value = ir_value

    def __repr__(self):
        return f"IrValueTensor(@{self.ir_value})"

    def resolve_ir_values(self, proc_trace: IrTrace) -> Sequence[Value]:
        return (self.ir_value,)
