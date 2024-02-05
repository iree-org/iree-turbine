# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
)

from contextlib import contextmanager

import torch

from ....support.ir_imports import (
    F32Type,
    F64Type,
    IndexType,
    IntegerType,
    IrType,
    Location,
    RankedTensorType,
    ShapedType,
    Value,
)

from ..ir_utils import (
    FunctionBuilder,
    ModuleBuilder,
)

from ..utils import (
    thread_state,
    tree_map,
)

ShapedTypeDynamicSizeSentinel = ShapedType.get_dynamic_size()

###############################################################################
# Tracing intrinsics
###############################################################################


class ProcedureTraceError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class IrTrace(FunctionBuilder):
    """Gets callbacks for tracing events."""

    __slots__ = []

    def finalize(self):
        """Called when the trace is finished (popped off the stack)."""
        pass

    def handle_call(self, target: "Intrinsic", args, kwargs):
        raise NotImplementedError(f"The current trace scope does not support calls")

    def handle_assignment(self, scope, target, updated_value):
        raise NotImplementedError(
            f"The current trace scope does not support assignment"
        )


def _trace_scopes() -> List[IrTrace]:
    try:
        trace_scopes = thread_state.trace_scopes
    except AttributeError:
        trace_scopes = thread_state.trace_scopes = []
    return trace_scopes


@contextmanager
def new_ir_trace_scope(ir_trace: IrTrace):
    trace_scopes = _trace_scopes()
    trace_scopes.append(ir_trace)
    try:
        yield ir_trace
    finally:
        ir_trace.finalize()
        del trace_scopes[-1]


def current_ir_trace() -> IrTrace:
    return _trace_scopes()[-1]


class Intrinsic:
    """Objects which interact natively with the tracing system implement this."""

    __slots__: List[str] = []

    def resolve_ir_values(self, proc_trace: "IrTrace") -> Sequence[Value]:
        raise NotImplementedError(
            f"Cannot use {self} as an expression in a procedural function"
        )

    def resolve_call(self, proc_trace: "IrTrace", *args, **kwargs):
        raise NotImplementedError(
            f"Cannot use {self} as the target of a call in a procedural function"
        )

    def resolve_assignment(self, proc_trace: "IrTrace", ir_values: Sequence[Value]):
        raise NotImplementedError(
            f"Cannot use {self} as the target of an assignment in a procedural function"
        )

    # Helpers for accessing the ir_value within the current trace.
    @property
    def ir_values(self) -> Sequence[Value]:
        return self.resolve_ir_values(current_ir_trace())

    @property
    def ir_value(self) -> Value:
        values = self.ir_values
        assert len(values) == 1, "Expected arity one intrinsic"
        return values[0]


class CallableIntrinsic(Intrinsic):
    """Intrinsic subclass that supports calls.

    This is separate so as to make error handling better (i.e. does not support
    calls) for intrinsics that are not callable.
    """

    __slots__ = []

    def __call__(self, *args, **kwargs):
        return current_ir_trace().handle_call(self, args, kwargs)


class AbstractIntrinsic:
    """Base class for descriptor types that can be converted to Python proxies."""

    __slots__: List[str] = []

    def create_intrinsic(self, value: Value) -> Intrinsic:
        """Creates a proxy object that can flow through a procedural trace."""
        raise NotImplementedError

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        """Gets the corresponding IR type."""
        raise NotImplementedError


###############################################################################
# Abstract types
###############################################################################


class AbstractTypedef:
    """Base class for instances which declare some form of public arg/result type definition."""

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        raise NotImplementedError


class Abstractifiable:
    """Indicates that a type knows how to abstractify itself."""

    def abstractify(self) -> AbstractTypedef:
        raise NotImplementedError


class TreeAbstractifiable:
    """Indicates that a type decomposes into a tree that can be abstractified."""

    def abstractify_tree(self) -> Any:
        raise NotImplementedError


class AbstractTensor(AbstractIntrinsic, AbstractTypedef):
    """Represents a tensor of known rank and dtype."""

    __slots__ = [
        "size",
        "dtype",
    ]

    def __init__(self, *size: Optional[int], dtype: torch.dtype = torch.float32):
        self.size = tuple(size)
        self.dtype = dtype

    def __repr__(self):
        return f"AbstractTensor({', '.join(str(s) for s in self.size)}, dtype={self.dtype})"

    def create_intrinsic(self, ir_value: Value) -> Intrinsic:
        return IrImmediateTensor(ir_value, self.dtype)

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        element_type = builder.torch_dtype_to_iree_type(self.dtype)
        with Location.unknown(builder.context):
            tensor_type = RankedTensorType.get(
                [
                    s if s is not None else ShapedTypeDynamicSizeSentinel
                    for s in self.size
                ],
                element_type,
            )
        return tensor_type


class AbstractScalar(AbstractIntrinsic, AbstractTypedef):
    """Represents a scalar value of some type."""

    __slots__ = [
        "label",
        "type_producer",
    ]

    def __init__(self, label: str, type_producer: Callable[[], IrType]):
        self.label = label
        self.type_producer = type_producer

    def __repr__(self):
        return f"AbstractScalar({self.label})"

    def create_intrinsic(self, ir_value: Value) -> Intrinsic:
        return IrImmediateScalar(ir_value)

    def get_ir_type(self, builder: ModuleBuilder) -> IrType:
        with builder.context:
            return self.type_producer()


# Concrete scalar types.
AbstractIndex = AbstractScalar("index", lambda: IndexType.get())
AbstractF32 = AbstractScalar("f32", lambda: F32Type.get())
AbstractF64 = AbstractScalar("f64", lambda: F64Type.get())
AbstractBool = AbstractScalar("bool", lambda: IntegerType.get_signless(1))
AbstractI32 = AbstractScalar("i32", lambda: IntegerType.get_signless(32))
AbstractI64 = AbstractScalar("i64", lambda: IntegerType.get_signless(64))


def abstractify_single_value(value) -> AbstractTypedef:
    if isinstance(value, AbstractTypedef):
        return value
    if isinstance(value, Abstractifiable):
        return value.abstractify()
    if isinstance(value, torch.Tensor):
        return AbstractTensor(*value.shape, dtype=value.dtype)
    raise TypeError(
        f"Cannot convert type {value.__class__} to an abstract type: {value}"
    )


def abstractify(tree):
    if isinstance(tree, TreeAbstractifiable):
        return tree.abstractify_tree()
    return tree_map(abstractify_single_value, tree)


# Circular iports.
from .primitives import (
    IrImmediateScalar,
    IrImmediateTensor,
)
