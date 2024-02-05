# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Concrete tracer for running buildable code.

from typing import (
    Any,
    Callable,
    List,
    Sequence,
)

from ....support.ir_imports import (
    Location,
    StringAttr,
    Value,
    func_d,
)

from ..ir_utils import (
    ModuleBuilder,
)

from ..utils import (
    logger,
    tree_flatten,
    tree_unflatten,
    treespec_dumps,
)

from .base import (
    AbstractIntrinsic,
    Intrinsic,
    IrTrace,
    ProcedureTraceError,
    new_ir_trace_scope,
)

from .globals import (
    LiveGlobalCollectionProxy,
)

###############################################################################
# Concrete procedure building IrTracer.
###############################################################################


class ProcedureTrace(IrTrace):
    """Captures execution of a Python func into IR."""

    __slots__ = [
        "proxy_posargs",
        "proxy_kwargs",
    ]

    def __init__(
        self,
        *,
        module_builder: ModuleBuilder,
        func_op: func_d.FuncOp,
        proxy_posargs,
        proxy_kwargs,
    ):
        super().__init__(module_builder=module_builder, func_op=func_op)
        self.proxy_posargs = proxy_posargs
        self.proxy_kwargs = proxy_kwargs

    @staticmethod
    def define_func(
        module_builder: ModuleBuilder,
        *,
        symbol_name: str,
        posargs: Sequence,
        kwargs: dict,
        loc: Location,
    ) -> "ProcedureTrace":
        # Unpack arguments.
        arguments_flat, arguments_tree_def = tree_flatten((posargs, kwargs))
        argument_ir_types = []
        for arg in arguments_flat:
            if not isinstance(arg, AbstractIntrinsic):
                raise ProcedureTraceError(f"Expected a AbstractIntrinsic but got {arg}")
            argument_ir_types.append(arg.get_ir_type(module_builder))

        with loc:
            _, func_op = module_builder.create_func_op(symbol_name, argument_ir_types)

        # Bind proxy arguments to an IR value.
        ir_proxy_arguments_flat = []
        for ir_value, arg_proxy_type in zip(
            func_op.body.blocks[0].arguments, arguments_flat
        ):
            ir_proxy_arguments_flat.append(arg_proxy_type.create_intrinsic(ir_value))

        # Unflatten.
        proxy_posargs, proxy_kwargs = tree_unflatten(
            ir_proxy_arguments_flat, arguments_tree_def
        )

        # Metadata.
        if arguments_flat:
            func_op.attributes["torch.args_schema"] = StringAttr.get(
                treespec_dumps(arguments_tree_def), context=module_builder.context
            )

        return ProcedureTrace(
            module_builder=module_builder,
            func_op=func_op,
            proxy_posargs=proxy_posargs,
            proxy_kwargs=proxy_kwargs,
        )

    def trace_py_func(self, py_f: Callable):
        with new_ir_trace_scope(self) as t:
            # TODO: Create IR proxies for python arguments.
            # See: https://github.com/nod-ai/SHARK-Turbine/issues/135
            return_py_value = _unproxy(py_f(*self.proxy_posargs, **self.proxy_kwargs))
            if return_py_value is None:
                self.emit_return()
            else:
                flat_return_py_values, schema = tree_flatten(return_py_value)
                flat_return_ir_values: List[Value] = []
                for py_value in flat_return_py_values:
                    flat_return_ir_values.extend(convert_py_value_to_ir(self, py_value))
                self.func_op.attributes["torch.return_schema"] = StringAttr.get(
                    treespec_dumps(schema), context=self.context
                )
                self.emit_return(*flat_return_ir_values)

    def handle_call(self, target: Intrinsic, args, kwargs):
        """Implements calls to jittable functions."""
        with self.loc, self.ip:
            return target.resolve_call(self, *args, **kwargs)

    def handle_assignment(self, scope, target, updated_value):
        logger.debug(
            "ASSIGN %r.%r = %r", scope.__class__, target.__class__, updated_value
        )
        self._recursive_assign(target, updated_value, set())

    def _recursive_assign(self, target, source, encountered_set):
        target = _unproxy(target)
        source = _unproxy(source)

        # Check for cycles.
        target_id = id(target)
        if target_id in encountered_set:
            raise TypeError(f"Cycle in tree assignment target")
        encountered_set.add(target_id)

        # Leaves/terminals.
        if isinstance(target, Intrinsic):
            if not isinstance(source, Intrinsic):
                raise TypeError(
                    f"Cannot assign mismatched leaf types in a tree: "
                    f"{target.__class__} vs {source.__class__}"
                )
            leaf_values = source.resolve_ir_values(self)
            target.resolve_assignment(self, leaf_values)
            return

        # Zip across dicts.
        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(
                    f"Mismatched dict assignment in a tree: {target.__class__} vs {source.__class__}"
                )
            target_keys = target.keys()
            source_keys = source.keys()
            if target_keys != source_keys:
                raise TypeError(
                    f"Mismatched dict keys in tree assignment: {target_keys} vs {source_keys}"
                )
            for k in target_keys:
                target_child = target[k]
                source_child = source[k]
                self._recursive_assign(target_child, source_child, encountered_set)
            return

        # Zip across lists/tuples (we let them be used interchangeably at the source).
        if isinstance(target, list):
            if not isinstance(source, (list, tuple)):
                if len(target) != len(source):
                    raise TypeError(
                        f"Mismatched sequence length in tree assignment: {len(target)} vs {len(source)}"
                    )
            for target_child, source_child in zip(target, source):
                self._recursive_assign(target_child, source_child, encountered_set)
            return

        raise TypeError(
            f"Cannot recursively assign through a container of {target.__class__}"
        )


def convert_py_value_to_ir(
    proc_trace: ProcedureTrace, py_value: Any
) -> Sequence[Value]:
    """Given procedurally traced python values, type check and convert to IR."""
    if isinstance(py_value, Intrinsic):
        return py_value.resolve_ir_values(proc_trace)
    if isinstance(py_value, Value):
        return [py_value]
    raise TypeError(
        f"Illegal type passed in procedural trace: {py_value.__class__} ({py_value})"
    )


def _unproxy(value):
    if isinstance(value, LiveGlobalCollectionProxy):
        return value._raw_collection
    return value
