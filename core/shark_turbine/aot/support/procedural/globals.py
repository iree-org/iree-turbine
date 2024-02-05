# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Global references in a module.

from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Sequence,
    Tuple,
)

import torch

from ....support.ir_imports import (
    IrType,
    Operation,
    Value,
    util_d,
)

from ..ir_utils import (
    GlobalAttributes,
    ModuleBuilder,
)

from ..utils import (
    TreeSpec,
    logger,
    tree_unflatten,
)

from .base import (
    AbstractScalar,
    AbstractTensor,
    Intrinsic,
    IrTrace,
    current_ir_trace,
)

from .primitives import (
    IrScalar,
    IrTensor,
)

###############################################################################
# Globals
###############################################################################


class LiveGlobalCollectionProxy:
    """Proxy object around a collection which knows how to redirect setitem."""

    __slots__ = ["_raw_collection"]

    def __init__(self, raw_collection):
        self._raw_collection = raw_collection

    def __getitem__(self, key: str):
        actual = self._raw_collection[key]
        if isinstance(actual, MaterializedGlobal):
            return actual
        else:
            return LiveGlobalCollectionProxy(actual)

    def __setitem__(self, key, value):
        item = self._raw_collection[key]
        if isinstance(item, MaterializedGlobal):
            current_ir_trace().handle_assignment(self, item, value)
        else:
            raise AttributeError(
                f"Globals collection {self._raw_collection.__class__} only supports assignment of leaves"
            )

    def __len__(self):
        return len(self._raw_collection)

    def __repr__(self):
        return f"LiveGlobalsProxy({self._raw_collection})"


class GlobalsDef:
    """Base class for all exporting descriptors."""

    __slots__ = [
        "_attrs",
    ]

    def __init__(self, attrs: GlobalAttributes):
        self._attrs = attrs

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Yields tuples of name/value exports."""
        raise NotImplementedError

    def schema(self) -> TreeSpec:
        """A schema used to unflatten for access from Python."""
        raise NotImplementedError

    def track(self, module_builder: ModuleBuilder, export_namespace: str) -> Any:
        """Track the given pack of globals, returning a struct that can be used to access them."""
        flat_globals = []
        for name, value in self.items():
            # Switch on types we support.
            fq_name = f"{export_namespace}.{name}"
            if isinstance(value, torch.Tensor):
                mapping = module_builder.global_ref_tracker.track(value)
                if not mapping.is_empty:
                    logger.debug(
                        "IGNORE EXISTING TRACKED TENSOR(%s): %r", fq_name, mapping
                    )
                    flat_globals.append(mapping.value)
                    continue
                (
                    actual_symbol_name,
                    global_op,
                    global_type,
                ) = module_builder.create_tensor_global(
                    f"_{fq_name}",
                    value,
                    attrs=self._attrs,
                    logical_name=fq_name,
                )
                mapping.value = IrGlobalTensor(
                    fq_name,
                    self,
                    symbol_name=actual_symbol_name,
                    global_op=global_op,
                    global_type=global_type,
                    dtype=value.dtype,
                )
                logger.debug("TRACK NEW TENSOR(%s): %r", fq_name, mapping)
                flat_globals.append(mapping.value)
                continue
            elif isinstance(value, AbstractTensor):
                global_type = value.get_ir_type(module_builder)
                (
                    actual_symbol_name,
                    global_op,
                ) = module_builder.create_typed_global(
                    f"_{fq_name}",
                    global_type,
                    attrs=self._attrs,
                    logical_name=fq_name,
                )
                flat_globals.append(
                    IrGlobalTensor(
                        fq_name,
                        self,
                        symbol_name=actual_symbol_name,
                        global_op=global_op,
                        global_type=global_type,
                        dtype=value.dtype,
                    )
                )
                continue
            elif isinstance(value, AbstractScalar):
                global_type = value.get_ir_type(module_builder)
                (
                    actual_symbol_name,
                    global_op,
                ) = module_builder.create_typed_global(
                    f"_{fq_name}",
                    global_type,
                    attrs=self._attrs,
                    logical_name=fq_name,
                )
                flat_globals.append(
                    IrGlobalScalar(
                        fq_name,
                        self,
                        symbol_name=actual_symbol_name,
                        global_op=global_op,
                        global_type=global_type,
                    )
                )
                continue

            raise TypeError(f"Unsupported global type: {value.__class__}")
        tree_globals = tree_unflatten(flat_globals, self.schema())
        if isinstance(tree_globals, MaterializedGlobal):
            return tree_globals
        else:
            return LiveGlobalCollectionProxy(tree_globals)


class MaterializedGlobal:
    """Tags an Ir* that is duck-typed as a global."""

    ...


class IrGlobalScalar(IrScalar, MaterializedGlobal):
    """An IrScalar that is loaded from a global and associated with its aggregate."""

    __slots__ = [
        "global_op",
        "global_type",
        "info",
        "export_name",
        "symbol_name",
    ]

    def __init__(
        self,
        export_name: str,
        info: GlobalsDef,
        *,
        symbol_name: str,
        global_op: Operation,
        global_type: IrType,
    ):
        super().__init__(global_type)
        self.info = info
        self.export_name = export_name
        self.symbol_name = symbol_name
        self.global_op = global_op

    def resolve_ir_values(self, trace: IrTrace) -> Sequence[Value]:
        with trace.loc, trace.ip:
            value = util_d.GlobalLoadOp(self.ir_type, self.symbol_name).result
        return [value]

    def resolve_assignment(self, proc_trace: "IrTrace", ir_values: Sequence[Value]):
        if len(ir_values) != 1:
            raise ValueError(
                f"Can only assign a single value to a global. Got {len(ir_values)}"
            )
        source_ir_type = ir_values[0].type
        if source_ir_type != self.ir_type:
            raise TypeError(
                f"Cannot assign to a global with a different type: {self.ir_type} != {source_ir_type}"
            )
        with proc_trace.loc, proc_trace.ip:
            util_d.GlobalStoreOp(ir_values[0], self.symbol_name)

    def set(self, other):
        t = current_ir_trace()
        self.resolve_assignment(t, super().set(other).ir_values)

    def __repr__(self):
        return (
            f"<IrGlobalScalar {self.export_name} = {self.symbol_name}:{self.ir_type}>"
        )


class IrGlobalTensor(IrTensor, MaterializedGlobal):
    """An IrScalar that is loaded from a global and associated with its aggregate."""

    __slots__ = [
        "global_op",
        "info",
        "export_name",
        "symbol_name",
    ]

    def __init__(
        self,
        export_name: str,
        info: GlobalsDef,
        *,
        symbol_name: str,
        global_op: Operation,
        global_type: IrType,
        dtype: torch.dtype,
    ):
        super().__init__(global_type, dtype)
        self.info = info
        self.export_name = export_name
        self.symbol_name = symbol_name
        self.global_op = global_op

    def resolve_ir_values(self, trace: IrTrace) -> Sequence[Value]:
        with trace.loc, trace.ip:
            value = util_d.GlobalLoadOp(self.ir_type, self.symbol_name).result
        return [value]

    def resolve_assignment(self, proc_trace: "IrTrace", ir_values: Sequence[Value]):
        if len(ir_values) != 1:
            raise ValueError(
                f"Can only assign a single value to a global. Got {len(ir_values)}"
            )
        source_ir_type = ir_values[0].type
        if source_ir_type != self.ir_type:
            raise TypeError(
                f"Cannot assign to a global with a different type: {self.ir_type} != {source_ir_type}"
            )
        with proc_trace.loc, proc_trace.ip:
            util_d.GlobalStoreOp(ir_values[0], self.symbol_name)

    def __repr__(self):
        return f"<MaterializedGlobal {self.export_name} = {self.symbol_name}:{self.ir_type}>"
