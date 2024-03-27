# Copyright 2024 Advanced Micro Devices, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Dict, List, Optional

import inspect

import torch

from torch.utils._pytree import (
    tree_flatten,
    tree_unflatten,
)

try:
    from torch.utils._pytree import treespec_pprint
except ImportError:
    # torch < 2.3 does not include this.
    treespec_pprint = lambda x: repr(x)  # type: ignore

from iree.compiler.extras.fx_importer import (
    FxImporter,
    FxImporterHooks,
    GraphNodeImporter,
)

from ....support.logging import aot_logger as logger

from ....support.ir_imports import (
    func_d,
    util_d,
    FlatSymbolRefAttr,
    FunctionType,
    IrType,
    Operation,
    StringAttr,
    TypeAttr,
    Value,
)

from ..ir_utils import (
    ModuleBuilder,
)

from .base import (
    CallableIntrinsic,
)

from .primitives import (
    IrImmediateTensor,
    IrTensor,
)

from .tracer import (
    IrTrace,
)


class ExportedProgramIntrinsic(CallableIntrinsic):
    def __init__(
        self,
        entry_func_op: Operation,
        entry_sig: torch.export.ModuleCallSignature,
        user_output_dtypes: List[Optional[torch.dtype]],
    ):
        self.entry_func_op = entry_func_op
        self.entry_sig = entry_sig
        self.user_output_dtypes = user_output_dtypes

    @property
    def function_type(self) -> FunctionType:
        return TypeAttr(self.entry_func_op.attributes["function_type"]).value

    @property
    def function_symbol(self) -> StringAttr:
        return StringAttr(self.entry_func_op.attributes["sym_name"])

    @property
    def function_visibility(self) -> StringAttr:
        return StringAttr(self.entry_func_op.attributes["sym_visibility"])

    def resolve_call(
        self,
        proc_trace: IrTrace,
        *py_args,
        **py_kwargs,
    ):
        visibility = self.function_visibility
        if visibility.value != "private":
            raise ValueError(
                f"Currently, only private ExportedPrograms can be called: "
                f"{self.function_symbol} is {visibility}"
            )

        # Flatten and convert py args to torch IR values by converting to
        # the canonical tree structure for args
        # (tuple of list of args, dict of kwargs).
        flat_py_args, args_tree = tree_flatten(((list(py_args),), py_kwargs))
        if args_tree != self.entry_sig.in_spec:
            raise ValueError(
                f"Mismatched arguments to exported program. \n"
                f"  Got: {treespec_pprint(args_tree)}\n"
                f"  Expected: {treespec_pprint(self.entry_sig.in_spec)} "
            )
        function_type = self.function_type
        flat_ir_args = [
            self._py_to_torch_ir(proc_trace, py_arg, torch_type)
            for py_arg, torch_type in zip(flat_py_args, function_type.inputs)
        ]

        # Call.
        with proc_trace.ip, proc_trace.loc:
            flat_ir_results = func_d.CallOp(
                function_type.results,
                FlatSymbolRefAttr.get(self.function_symbol.value),
                flat_ir_args,
            ).results

        # Convert torch IR values to python.
        flat_py_results = [
            self._torch_ir_to_py(proc_trace, ir_value, dtype)
            for ir_value, dtype in zip(flat_ir_results, self.user_output_dtypes)
        ]

        return tree_unflatten(flat_py_results, self.entry_sig.out_spec)

    def _py_to_torch_ir(
        self, proc_trace: IrTrace, py_value, torch_type: IrType
    ) -> Value:
        type_converter = proc_trace.module_builder.native_type_converter
        if isinstance(py_value, IrTensor):
            # TODO: Allow certain static info casts.
            return type_converter.materialize_native_to_torch(
                py_value.ir_value, torch_type
            )
        else:
            raise ValueError(
                f"Unsupported type in arguments of call to ExportedProgram: "
                f"{type(py_value)}: {py_value}"
            )

    def _torch_ir_to_py(
        self, proc_trace: IrTrace, ir_value: Value, dtype: Optional[torch.dtype]
    ):
        type_converter = proc_trace.module_builder.native_type_converter
        native_ir_value = type_converter.materialize_torch_to_native(ir_value)
        if dtype is not None:
            return IrImmediateTensor(native_ir_value, dtype)
        else:
            raise TypeError(
                f"Unknown PyTorch->IREE value mapping for ExportedProgram output: "
                f"{native_ir_value}"
            )


def import_exported_program(
    module_builder: ModuleBuilder,
    exported_program: torch.export.ExportedProgram,
    symbol_name: str,
    symbol_visibility: Optional[str],
) -> ExportedProgramIntrinsic:
    fx_importer = _create_fx_importer(module_builder)
    entry_func_op = fx_importer.import_program(
        exported_program, func_name=symbol_name, func_visibility=symbol_visibility
    )

    module_call_graph = exported_program.module_call_graph
    assert len(module_call_graph) >= 1, "Expected at least one module call signature"
    entry_module_call_entry = module_call_graph[0]
    assert (
        entry_module_call_entry.fqn == ""
    ), "Expected first module call entry to be unnamed"

    # We want additional torch-level metadata about any user outputs.
    # This will help us create a true python fake without loss of information.
    # TODO: It is unclear how much switchiness is actually needed here as
    # modern use is pretty constrained. Potentially streamline the body of
    # the for loop once done with full test cases available.
    user_output_dtypes: list[Optional[torch.dtype]] = []
    node_map: Dict[str, torch.fx.Node] = {
        n.name: n for n in exported_program.graph.nodes
    }
    for user_output in exported_program.graph_signature.user_outputs:
        output_node = node_map[user_output]
        tensor_meta = output_node.meta.get("tensor_meta")
        fake_val = output_node.meta.get("val")
        dtype = None
        if tensor_meta is not None:
            dtype = tensor_meta.dtype
        elif fake_val is not None:
            dtype = fake_val.dtype
        user_output_dtypes.append(dtype)

    return ExportedProgramIntrinsic(
        entry_func_op, entry_module_call_entry.signature, user_output_dtypes
    )


class _Hooks(FxImporterHooks):
    def __init__(self, module_builder: ModuleBuilder):
        self.module_builder = module_builder

    def resolve_literal(self, gni: GraphNodeImporter, literal: Any) -> Optional[Value]:
        module_builder = self.module_builder

        # We support resolution of tracked reference types. Currently this
        # only includes Tensors. All others we let the importer do what it
        # is going to do.
        if not isinstance(literal, torch.Tensor):
            return None

        # See if we know about it.
        mapping = module_builder.global_ref_tracker.track(literal)
        if mapping.is_empty:
            # If it is unknown, just let the default importer take it on.
            return None

        # Already materialized.
        logger.debug("Resolved defined global for literal %r", mapping)
        materialized_global: MaterializedGlobal = mapping.value  # type: ignore

        # Emit a global load and conversion.
        vtensor_type = gni._cc.tensor_to_vtensor_type(literal)
        loaded_value = util_d.GlobalLoadOp(
            materialized_global.ir_type, materialized_global.symbol_name
        ).result
        converted_value = Operation.create(
            "torch_c.from_builtin_tensor",
            results=[vtensor_type],
            operands=[loaded_value],
        ).result
        return converted_value


# In https://github.com/llvm/torch-mlir/pull/3046, the FxImporter was
# extended to accept a "module_op" as an Operation (vs a Module). Switch for
# compatibility.
_fx_importer_accepts_module_op = (
    "module_op" in inspect.getfullargspec(FxImporter).kwonlyargs
)


def _create_fx_importer(module_builder: ModuleBuilder) -> FxImporter:
    hooks = _Hooks(module_builder)
    if _fx_importer_accepts_module_op:
        # New path.
        return FxImporter(
            module_op=module_builder.module_op,
            config_check=False,
            py_attr_tracker=module_builder.fx_py_attr_tracker,
            hooks=hooks,
        )
    else:
        # Legacy path.
        class FakeModule:
            def __init__(self, op):
                self._op = module_builder.module_op

            @property
            def context(self):
                return self._op.context

            @property
            def operation(self):
                return self._op

            @property
            def body(self):
                return self._op.regions[0].blocks[0]

        return FxImporter(
            module=FakeModule(module_builder.module_op),
            config_check=False,
            py_attr_tracker=module_builder.fx_py_attr_tracker,
            hooks=hooks,
        )
