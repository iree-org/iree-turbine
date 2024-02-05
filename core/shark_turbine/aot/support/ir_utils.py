# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Generator, List, Optional, Sequence, Tuple

from pathlib import Path
import tempfile

import numpy as np
import torch

from ...importers.fx_importer import (
    ContextCache,
)

from ...importers.utils import (
    RefTracker as FxRefTracker,
)

from ...dynamo.type_conversion import (
    NativeTypeConverter,
)

from ...support.ir_imports import (
    Attribute,
    BF16Type,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MLIRError,
    Operation,
    RankedTensorType,
    StringAttr,
    SymbolTable,
    TypeAttr,
    UnitAttr,
    Value,
    arith_d,
    func_d,
    tensor_d,
)

from ...support.conversions import (
    TORCH_DTYPE_TO_IREE_TYPE,
)

from .utils import (
    RefTracker,
    logger,
)

###############################################################################
# Configuration
###############################################################################

# Maps a name to an altered name. If returns None, then the original
# name is used (this lets dict.get serve as a NameMapCallback).
NameMapCallback = Callable[[str], Optional[str]]


class GlobalAttributes:
    """Settings for how to initialize the global."""

    __slots__ = [
        "mutable",
        "external",
        "external_scope",
        "name_mapper",
        "noinline",
        "uninitialized",
    ]

    def __init__(
        self,
        mutable: bool = False,
        external: Optional[bool] = None,
        external_scope: Optional[str] = None,
        name_mapper: Optional[NameMapCallback] = None,
        noinline: bool = True,
        uninitialized: Optional[bool] = None,
    ):
        if external and uninitialized:
            raise ValueError(
                f"Globals with external=True cannot also have uninitialized=True"
            )
        if uninitialized and not mutable:
            raise ValueError(
                f"Globals with uninitialized=True must also be mutable=True"
            )
        self.mutable = mutable
        self.external = external
        self.external_scope = external_scope
        self.name_mapper = name_mapper
        self.noinline = noinline
        self.uninitialized = uninitialized

    def map_name(self, name: str) -> str:
        if self.name_mapper:
            new_name = self.name_mapper(name)
            if new_name is not None:
                return new_name
        return name


###############################################################################
# Builders
###############################################################################


class ModuleBuilder:
    """Wrapper around module and IR accounting for a module being built."""

    __slots__ = [
        "body",
        "cache",
        "context",
        "fx_py_attr_tracker",
        "global_ip",
        "ip",
        "module_op",
        "symbol_table",
        "global_ref_tracker",
        "native_type_converter",
    ]

    def __init__(self, module_op: Operation):
        self.module_op = module_op
        self.context = module_op.context
        self.body = module_op.regions[0].blocks[0]
        self.symbol_table = SymbolTable(module_op)
        self.global_ip = InsertionPoint.at_block_begin(self.body)
        self.ip = InsertionPoint(self.body)
        self.cache = ContextCache(self.context)
        # Tracks global references to a MaterializedGlobal.
        self.global_ref_tracker = RefTracker()
        # Usually the FxImporter makes a new ref tracker for each invocation,
        # but we want to preserve it across individual JIT evaluations so
        # as to better intern tensors to attributes.
        self.fx_py_attr_tracker = FxRefTracker()
        self.native_type_converter = NativeTypeConverter(self.context)

    def handle_mlir_error(self, op: Operation, e: MLIRError, message: str):
        # TODO: Replace with a real dumping facility.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/136
        dump_path = Path(tempfile.gettempdir()) / "turbine_module_builder_error.mlir"
        logger.exception(f"{message} (dumping to {dump_path})")
        try:
            with open(dump_path, "wb") as f:
                op.print(
                    f,
                    binary=True,
                    print_generic_op_form=True,
                    large_elements_limit=100,
                )
            logger.debug(f"Dump complete to {dump_path}")
        except Exception:
            logger.exception("Error generating dump file")

    def finalize_construct(self):
        try:
            self.module_op.verify()
        except MLIRError as e:
            self.handle_mlir_error(self.module_op, e, "module failed to verify")
            raise

    def create_func_op(
        self,
        symbol_name: str,
        argument_types: Sequence[IrType],
        is_public: bool = True,
        add_entry_block: bool = True,
    ) -> Tuple[str, func_d.FuncOp]:
        with self.ip:
            ftype = FunctionType.get(argument_types, [])
            func_op = func_d.FuncOp(symbol_name, ftype)
            if not is_public:
                func_op.attributes["sym_visibility"] = StringAttr.get("private")
            if add_entry_block:
                func_op.add_entry_block()
            self.symbol_table.insert(func_op)
            actual_symbol_name = StringAttr(func_op.attributes["sym_name"]).value
            return actual_symbol_name, func_op

    def torch_dtype_to_iree_type(self, dtype: torch.dtype) -> IrType:
        try:
            with self.context:
                return TORCH_DTYPE_TO_IREE_TYPE[dtype]()
        except KeyError:
            raise TypeError(f"Could not map Torch dtype {dtype} to an IREE type")

    def create_tensor_global(
        self,
        symbol_name: str,
        t: torch.Tensor,
        *,
        attrs: GlobalAttributes,
        logical_name: Optional[str] = None,
    ) -> Tuple[str, Operation, IrType]:
        element_type = self.torch_dtype_to_iree_type(t.dtype)
        with self.global_ip, Location.unknown():
            tensor_type = RankedTensorType.get(list(t.shape), element_type)
            ir_attrs = {
                "sym_name": StringAttr.get(symbol_name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(tensor_type),
            }
            if attrs.noinline:
                ir_attrs["noinline"] = UnitAttr.get()
            if attrs.mutable:
                ir_attrs["is_mutable"] = UnitAttr.get()
            if attrs.external:
                # Emit named external reference.
                external_scope_attr = StringAttr.get(attrs.external_scope or "model")
                external_name = attrs.map_name(
                    logical_name if logical_name is not None else symbol_name
                )
                external_name_attr = StringAttr.get(external_name)
                # TODO: Have real Python builders for this.
                ir_attrs["initial_value"] = Attribute.parse(
                    f"#stream.parameter.named<{external_scope_attr}::{external_name_attr}> : {tensor_type}"
                )
            elif attrs.uninitialized:
                # Emit unitialized initial_value to signal that the memory
                # is valid but has undefined contents.
                # TODO: Have real Python builders for this.
                ir_attrs["initial_value"] = Attribute.parse(
                    f"#util.uninitialized : {tensor_type}"
                )
            else:
                # Emit inline initialized.
                detached_tensor = t.detach().contiguous().cpu()
                array = np.array(detached_tensor)
                # We know that a Numpy array is a ReadableBuffer so ignore type error.
                contents = memoryview(array)  # type: ignore
                shape_desc = "_".join([str(d) for d in t.shape])
                blob_name = f"torch_tensor_{shape_desc}_{str(t.dtype)}"
                elements_attr = DenseResourceElementsAttr.get_from_buffer(
                    contents, blob_name, tensor_type
                )
                ir_attrs["initial_value"] = elements_attr

            global_op = Operation.create("util.global", attributes=ir_attrs)
            self.symbol_table.insert(global_op)
            actual_symbol_name = StringAttr(global_op.attributes["sym_name"]).value
            return actual_symbol_name, global_op, tensor_type

    def create_typed_global(
        self,
        symbol_name: str,
        global_type: IrType,
        *,
        attrs: GlobalAttributes,
        logical_name: Optional[str] = None,
    ) -> Tuple[str, Operation]:
        with self.global_ip, Location.unknown():
            ir_attrs = {
                "sym_name": StringAttr.get(symbol_name),
                "sym_visibility": StringAttr.get("private"),
                "type": TypeAttr.get(global_type),
            }
            if attrs.noinline:
                ir_attrs["noinline"] = UnitAttr.get()
            if attrs.mutable:
                ir_attrs["is_mutable"] = UnitAttr.get()
            if attrs.uninitialized:
                # Emit unitialized initial_value to signal that the memory
                # is valid but has undefined contents.
                # TODO: Have real Python builders for this.
                ir_attrs["initial_value"] = Attribute.parse(
                    f"#util.uninitialized : {global_type}"
                )
            else:
                # Initialized by default.
                ir_attrs["initial_value"] = self._create_initial_value_for_type(
                    global_type
                )
            global_op = Operation.create("util.global", attributes=ir_attrs)
            self.symbol_table.insert(global_op)
            actual_symbol_name = StringAttr(global_op.attributes["sym_name"]).value
            return actual_symbol_name, global_op

    def _create_initial_value_for_type(self, t: IrType) -> Attribute:
        # TODO(#169): Implement something upstream for this (it exists in the C++ API)
        # and use it.
        if RankedTensorType.isinstance(t):
            rtt = RankedTensorType(t)
            if not rtt.has_static_shape:
                raise ValueError(
                    "Cannot create initialization value for dynamic shaped tensor"
                )
            element_attr = self._create_initial_value_for_type(rtt.element_type)
            return DenseElementsAttr.get_splat(t, element_attr)
        elif IntegerType.isinstance(t):
            return IntegerAttr.get(t, 0)
        elif F32Type.isinstance(t) or F64Type.isinstance(t) or F16Type.isinstance(t):
            # TODO(#170): There should be a common way to check if a FloatType.
            return FloatAttr.get(t, 0.0)
        elif IndexType.isinstance(t):
            return IntegerAttr.get(IndexType.get(), 0)
        else:
            raise ValueError(
                f"Cannot create a default initialization value for type {t}"
            )


class FunctionBuilder:
    """Helpers for building function bodies."""

    __slots__ = [
        "module_builder",
        "func_op",
        "context",
        "ip",
        "return_types",
        "loc",
    ]

    def __init__(
        self,
        *,
        module_builder: ModuleBuilder,
        func_op: func_d.FuncOp,
    ):
        self.module_builder = module_builder
        self.func_op = func_op
        self.context = func_op.context
        self.ip = InsertionPoint(self.func_op.entry_block)
        self.return_types: Optional[Sequence[IrType]] = None
        self.loc = self.func_op.location

    def emit_return(self, *ir_values: Value):
        with self.loc, self.ip:
            func_d.ReturnOp(ir_values)
            # Check or rewrite the function return type.
            value_types = [v.type for v in ir_values]
            if self.return_types:
                if value_types != self.return_types:
                    raise ValueError(
                        f"Multi-return function must return same types. "
                        f"{value_types} vs {self.return_types}"
                    )
                return
            self.return_types = value_types
            ftype = self.func_op.type
            ftype = FunctionType.get(ftype.inputs, value_types)
            self.func_op.attributes["function_type"] = TypeAttr.get(ftype)
            try:
                self.func_op.verify()
            except MLIRError as e:
                self.module_builder.handle_mlir_error(
                    self.func_op, e, "created function does not verify"
                )
                raise


###############################################################################
# Helpers
###############################################################################


def build_index_attribute(value: int) -> IntegerAttr:
    return IntegerAttr.get(IndexType.get(), value)


def build_index_value(
    value: int, constant_cache: Optional[dict[int, Value]] = None
) -> Value:
    if constant_cache is not None and value in constant_cache:
        return constant_cache[value]
    index_value = arith_d.ConstantOp(IndexType.get(), value).result
    if constant_cache is not None:
        constant_cache[value] = index_value
    return index_value


def build_tensor_dim_value(
    t: Value, dim: int, constant_cache: Optional[dict[int, Value]] = None
) -> Value:
    dim_value = build_index_value(dim, constant_cache=constant_cache)
    return tensor_d.DimOp(t, dim_value).result


# API name  inspired by mlir/python/mlir/dialects/_arith_ops_ext.py
def _is_float_type(type):
    return isinstance(type, (BF16Type, F16Type, F32Type, F64Type))


def _is_integer_like_type(type):
    return isinstance(type, (IntegerType, IndexType))
