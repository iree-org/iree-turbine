# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import enum
import inspect
import logging
from pathlib import Path
import re
import weakref
import sys

from . import builtins

from ..support.ir_imports import (
    Context,
    Location,
    MLIRError,
    Module,
    Operation,
    PassManager,
    StringAttr,
)
from ..transforms.general.custom_op_expansion import ExpandCustomOpsPass

from .support.procedural import (
    GlobalsDef,
    ProcedureTrace,
    current_ir_trace,
)

from .support.ir_utils import (
    ModuleBuilder,
)

from .support.utils import (
    logger,
)

__all__ = [
    "CompiledModule",
]

################################################################################
# Data structures
################################################################################


class ImportPhase(enum.IntEnum):
    # Imports to torch dialect IR.
    TORCH_IR = 0

    # Performs custom op expansion and post processing for known custom ops.
    CUSTOM_OP_EXPANSION = 1

    # Compiles to valid MLIR that IREE can ingest as an input with the
    # input-type of torch.
    IMPORT = CUSTOM_OP_EXPANSION

    # Runs the IREE input pipeline to compile to internal form.
    IREE_INTERNAL = 2

    # The full import pipeline (this is an alias for another enum value).
    FULL = IREE_INTERNAL

    @staticmethod
    def parse(spec: Union[str, None, "ImportPhase"]) -> "ImportPhase":
        if spec is None:
            return ImportPhase.IMPORT
        if isinstance(spec, ImportPhase):
            return spec
        spec = spec.upper().replace("-", "_")
        if spec not in ImportPhase.__members__:
            raise ValueError(
                f"For import_phase= argument, expected one of: "
                f"{', '.join(ImportPhase.__members__.keys())}"
            )
        return ImportPhase[spec]

    def __str__(self):
        return self.name


class PyOnlyDef:
    """Exportable that does not export but can be resolved in Python."""

    __slots__ = ["py_value"]

    def __init__(self, py_value):
        self.py_value = py_value

    def __str__(self):
        return str(self.py_value)

    def __repr__(self):
        return repr(self.py_value)

    def __call__(self, *args, **kwargs):
        return self.py_value(*args, **kwargs)


class ExportProcDef:
    __slots__ = [
        "callable",
        "export_name",
        "signature",
        "file_line_loc",
    ]

    def __init__(
        self,
        export_name: str,
        callable: Callable,
        *,
        signature,
        file_line_loc: Optional[Tuple[str, int]] = None,
    ):
        self.export_name = export_name
        self.callable = callable
        self.signature = signature
        self.file_line_loc = file_line_loc

    def copy(self) -> "ExportProcDef":
        return ExportProcDef(self.export_name, self.callable, signature=self.signature)

    def __repr__(self):
        return f"<def {self.export_name}({self.signature})>"


Exportable = Union[ExportProcDef, PyOnlyDef, GlobalsDef]


class CompiledModuleClassInfo:
    __slots__ = [
        "all_exports",
        "ir_module_name",
    ]

    def __init__(self, *, ir_module_name: str):
        self.ir_module_name = ir_module_name
        self.all_exports: Dict[str, Exportable] = dict()

    def add_export(self, key: str, value: Exportable):
        if key in self.all_exports:
            raise TypeError(f"Cannot export attribute more than once: {key}")
        self.all_exports[key] = value

    @property
    def export_procs(self) -> Generator[Tuple[str, ExportProcDef], None, None]:
        return filter(
            lambda kv_tuple: isinstance(kv_tuple[1], ExportProcDef),
            self.all_exports.items(),
        )

    @property
    def py_only_defs(self) -> Generator[Tuple[str, PyOnlyDef], None, None]:
        return filter(
            lambda kv_tuple: isinstance(kv_tuple[1], PyOnlyDef),
            self.all_exports.items(),
        )

    @property
    def globals_defs(self) -> Generator[Tuple[str, GlobalsDef], None, None]:
        return filter(
            lambda kv_tuple: isinstance(kv_tuple[1], GlobalsDef),
            self.all_exports.items(),
        )

    def def_attribute(self, key, value):
        # Some decorators, the only thing we do is convert them to PyOnlyDef.
        # Do that first so the generic descriptor code below handles them.
        if isinstance(value, builtins.jittable):
            value = PyOnlyDef(value)

        # Detect our own descriptors.
        if isinstance(value, GlobalsDef):
            logging.debug("DEFINE GLOBALS: %s = %r", key, value)
            self.add_export(key, value)
            return value
        if isinstance(value, ExportProcDef):
            value = value.copy()
            if value.export_name is None:
                value.export_name = key
            self.add_export(key, value)
            return value

        if isinstance(value, PyOnlyDef):
            logging.debug("DEFINE PY_ONLY: %s = %r", key, value)
            self.add_export(key, value)
            return value

        # Infer if it is an exported function.
        if callable(value) and inspect.isfunction(value):
            return self.def_export_proc(key, value)

        raise TypeError(
            f"cannot set arbitrary Python value '{key}' on "
            f"compiled module: {value!r}"
        )

    def def_export_proc(self, name, f) -> ExportProcDef:
        logging.debug("DEFINE EXPORT: %s = %r", name, f)
        # Get a reasonable location.
        file_line_loc = None
        try:
            sourcefile = inspect.getsourcefile(f)
            _, linenums = sourcelines = inspect.getsourcelines(f)
            if sourcefile and linenums:
                file_line_loc = [sourcefile, linenums[0]]
        except TypeError:
            pass

        sig = inspect.signature(f)
        if len(sig.parameters) < 1:
            raise TypeError(
                f"export proc '{name}' is expected to have a 'self' parameter"
            )

        # By default, we discover signature details from default values
        # on the function. But we should also source from an annotation.
        input_sig = []
        parameter_list = list(sig.parameters.values())
        # TODO: Reconstitute a pytree so as to handle kwargs?
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/128
        for param in parameter_list[1:]:
            if (
                param.kind != inspect.Parameter.POSITIONAL_ONLY
                and param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                raise TypeError(
                    f"exported functions only support positional parameters"
                )
            param_desc = param.default
            if param_desc is inspect.Parameter.empty:
                # TODO: Merge from a decorator?
                # See: https://github.com/nod-ai/SHARK-Turbine/issues/126
                raise TypeError(
                    f"export function {name} missing required default value annotation "
                    f"for '{param.name}'"
                )
            input_sig.append(param_desc)

        info = ExportProcDef(name, f, signature=input_sig, file_line_loc=file_line_loc)
        self.add_export(name, info)
        return info


class CompiledModuleInstanceInfo:
    """Info class for compiled module instances."""

    __slots__ = [
        "class_info",
        "module_builder",
        "shadow_dict",
        "current_import_phase",
    ]

    def __init__(
        self, class_info: CompiledModuleClassInfo, module_builder: ModuleBuilder
    ):
        self.class_info = class_info
        self.module_builder = module_builder
        # The shadow dict holds instance attributes. We stash them here and the
        # Program instance itself arbitrates access via getattr/setattr.
        self.shadow_dict = dict()
        self.current_import_phase = ImportPhase.TORCH_IR


################################################################################
# Live reference accounting
################################################################################

_all_compiled_module_class_infos: Dict[
    "CompiledModuleMeta", CompiledModuleClassInfo
] = weakref.WeakKeyDictionary()

_all_compiled_module_instance_infos: Dict[
    "CompiledModule", CompiledModuleInstanceInfo
] = weakref.WeakKeyDictionary()


################################################################################
# CompiledModule and metaclass
################################################################################

# Gate that is set to True once metaclass setup is complete.
_metaclass_setup_complete = False


@property
def _blackhole_instance_attribute(self):
    # We're not here.
    raise AttributeError


def _uncallable_public_export(*args, **kwargs):
    raise RuntimeError(f"Calls to exported functions not yet supported")


_COMPILED_MODULE_API_ATTRIBUTES = [
    "expand_custom_ops",
    "export_global",
    "get_class_info",
    "get_info",
    "get_module_builder",
    "get_mlir_module",
    "jittable",
    "run_import",
    "run_pass_pipeline",
    "save_mlir",
]


class CompiledModuleMeta(type):
    """Metaclass for all CompiledModule subclasses.

    Do not use directly.
    """

    # __new__ on a metaclass is called when a new subclass is constructed.
    # It is passed the dictionary of declared attributes and any keyword
    # arguments from the class declaration:
    #   class Foo(Bar, kwarg="you probably just learned this is possible"):
    def __new__(mcls, name: str, bases, dct, *, export_name: Optional[str] = None):
        if not _metaclass_setup_complete:
            return type.__new__(mcls, name, bases, dct)

        ir_module_name = _derive_ir_module_name(name, export_name)
        logger.debug("Create new CompiledModule: %s", ir_module_name)
        info = CompiledModuleClassInfo(ir_module_name=ir_module_name)

        # Process that attributes that were set as part of class definition.
        # Any attributes that we decide are part of the compiled module
        # are removed and appropriately transferred to the backing info
        # hierarchy.
        del_attr_keys = set()
        for key, value in dct.items():
            if key.startswith("__") and key.endswith("__"):
                continue
            del_attr_keys.add(key)
            info.def_attribute(key, value)

        for key in del_attr_keys:
            del dct[key]

        # The CompiledModule exports a number of its own API methods, which
        # we explicitly hide on subclasses and instances.
        for key in _COMPILED_MODULE_API_ATTRIBUTES:
            if key not in dct:
                dct[key] = _blackhole_instance_attribute

        # Inheriting methods, globals, and export from parent class.
        # Use case such as building a child-class to StatelessLlama.
        for base in bases:
            if base is CompiledModule:
                continue
            base_exports = _all_compiled_module_class_infos[base].all_exports
            for export_name in base_exports:
                if export_name in info.all_exports:
                    continue
                info.all_exports[export_name] = base_exports[export_name]

        # Finish construction.
        new_class = type.__new__(mcls, name, bases, dct)
        _all_compiled_module_class_infos[new_class] = info
        return new_class

    # Gets unresolved attributes on classes of this meta-class.
    def __getattr__(cls, key):
        # CompiledModule does not expose anything else.
        if cls is CompiledModule:
            raise AttributeError(f"CompiledModule.{key}")
        info = CompiledModule.get_class_info(cls)
        try:
            return info.all_exports[key]
        except KeyError:
            raise AttributeError


class CompiledModule(metaclass=CompiledModuleMeta):
    """Base class for all staged modules."""

    @staticmethod
    def get_class_info(cls: CompiledModuleMeta) -> CompiledModuleClassInfo:
        return _all_compiled_module_class_infos[cls]

    @staticmethod
    def get_info(inst: "CompiledModule") -> CompiledModuleInstanceInfo:
        return _all_compiled_module_instance_infos[inst]

    @staticmethod
    def get_module_builder(inst: "CompiledModule") -> Operation:
        if not isinstance(inst, CompiledModule):
            raise ValueError(
                f"Expected a CompiledModule instance but got: {inst.__class__}"
            )
        info = CompiledModule.get_info(inst)
        return info.module_builder

    @staticmethod
    def get_mlir_module(inst: "CompiledModule") -> Operation:
        return CompiledModule.get_module_builder(inst).module_op

    @staticmethod
    def run_import(
        inst: "CompiledModule", import_to: Union[ImportPhase, str, None] = "import"
    ):
        import_to = ImportPhase.parse(import_to)
        info = CompiledModule.get_info(inst)
        for phase in [
            ImportPhase.TORCH_IR,
            ImportPhase.CUSTOM_OP_EXPANSION,
            ImportPhase.IREE_INTERNAL,
        ]:
            if phase > import_to:
                logger.debug("Stopped import at phase %s", info.current_import_phase)
                break
            if info.current_import_phase >= phase:
                continue
            logger.debug("Run import phase %s", phase)
            if phase == ImportPhase.TORCH_IR:
                # Starting phase. Do nothing.
                ...
            elif phase == ImportPhase.CUSTOM_OP_EXPANSION:
                CompiledModule.expand_custom_ops(inst)
            elif phase == ImportPhase.IREE_INTERNAL:
                CompiledModule.run_pass_pipeline(inst, "builtin.module(torch-to-iree)")
            else:
                assert False, f"Phase {phase} not handled in switch"
            info.current_import_phase = phase

    @staticmethod
    def expand_custom_ops(inst: "CompiledModule"):
        """Performs custom torch.operator expansion for known custom ops."""
        logger.debug("Expand known torch.operator custom ops")
        module_op = CompiledModule.get_mlir_module(inst)
        p = ExpandCustomOpsPass(module_op)
        p.run()

    @staticmethod
    def run_pass_pipeline(
        inst: "CompiledModule", pipeline: str, enable_ir_printing: bool = False
    ):
        """Runs an arbitrary pass pipeline against the current IR.

        Args:
          pipeline: The text format pass pipeline as supported by PassManager.parse.
          enable_ir_printing: Enables print-after-all to stderr.
        """
        logger.debug("Run pass pipeline: %s", pipeline)
        module_op = CompiledModule.get_mlir_module(inst)
        with module_op.context:
            pm = PassManager.parse(pipeline)
            if enable_ir_printing:
                module_op.context.enable_multithreading(False)
                pm.enable_ir_printing()
            try:
                pm.run(module_op)
            except MLIRError:
                # TODO: Better error handling.
                # See: https://github.com/nod-ai/SHARK-Turbine/issues/127
                print(module_op, file=sys.stderr)
                raise

    @staticmethod
    def save_mlir(inst: "CompiledModule", path: Union[Path, str]):
        """Saves a snapshot of the MLIR module in this CompiledModule to a file.

        This is a convenience wrapper around the facilities of the underlying
        API and does not expose all features.

        Args:
          path: The file path to write to. If the extension is ".mlirbc", it
            will be written as bytecode.
        """
        path = Path(path)
        bytecode = path.suffix == ".mlirbc"
        module_op = CompiledModule.get_mlir_module(inst)
        with open(path, "wb") as f:
            if bytecode:
                module_op.write_bytecode(f)
            else:
                module_op.print(f, binary=True)

    jittable = staticmethod(builtins.jittable)

    def __getattr__(self, name):
        info = CompiledModule.get_info(self)
        try:
            return info.shadow_dict[name]
        except KeyError:
            raise AttributeError(f"Attribute {name} not defined")

    def __setattr__(self, name, value):
        info = CompiledModule.get_info(self)
        try:
            descriptor = info.shadow_dict[name]
        except KeyError:
            raise AttributeError(f"Attribute {name} cannot be set")
        current_ir_trace().handle_assignment(self, descriptor, value)

    def __new__(
        cls,
        *,
        context: Optional[Context] = None,
        module_op: Optional[Operation] = None,
        import_to: Union[ImportPhase, None, str] = "import",
    ):
        import_to = ImportPhase.parse(import_to)
        self = super().__new__(cls)
        class_info = CompiledModule.get_class_info(cls)
        if context and module_op:
            raise ValueError("Only one of context= or module_op= can be specified")
        if not context and not module_op:
            try:
                context = Context.current
            except ValueError:
                pass

        if not context:
            context = Context()

        if not module_op:
            with context:
                loc = Location.unknown(context=context)
                module = Module.create(loc)
                module_op = module.operation
                module_op.attributes["sym_name"] = StringAttr.get(
                    class_info.ir_module_name, context=context
                )
        module_builder = ModuleBuilder(module_op)
        info = CompiledModuleInstanceInfo(class_info, module_builder=module_builder)
        _all_compiled_module_instance_infos[self] = info

        # Instantiate globals
        for key, globals_def in info.class_info.globals_defs:
            info.shadow_dict[key] = globals_def.track(module_builder, key)

        # Make PyOnly defs visible.
        for key, py_def in info.class_info.py_only_defs:
            info.shadow_dict[key] = py_def.py_value

        # Instantiate procs.
        # TODO: This should be done in two phases, first binding the symbols
        # and then defining them, enabling dependence.
        # See: https://github.com/nod-ai/SHARK-Turbine/issues/129
        for key, proc_def in info.class_info.export_procs:

            def do_export(proc_def: ExportProcDef):
                def invoke_with_self(*args, **kwargs):
                    return proc_def.callable(self, *args, **kwargs)

                logger.debug("Generating procedural function: %s", key)
                if proc_def.file_line_loc:
                    loc = Location.file(
                        proc_def.file_line_loc[0],
                        proc_def.file_line_loc[1],
                        col=0,
                        context=module_builder.context,
                    )
                else:
                    loc = Location.unknown(context=module_builder.context)
                trace = ProcedureTrace.define_func(
                    module_builder,
                    symbol_name=proc_def.export_name,
                    posargs=proc_def.signature,
                    kwargs={},  # TODO(#128): kwargs
                    loc=loc,
                )
                trace.trace_py_func(invoke_with_self)
                info.shadow_dict[key] = _uncallable_public_export

            do_export(proc_def)

        module_builder.finalize_construct()
        CompiledModule.run_import(self, import_to)
        return self


_metaclass_setup_complete = True

################################################################################
# Utilities
################################################################################


def _derive_ir_module_name(class_name: str, explicit_name: Optional[str]):
    """Returns an appropriate module export name given a class name and override.

    If an explicit_name is given, that is used as is. Otherwise, the class name
    is mangled by:
      * Removing and "Module" suffix.
      * Converting camel case to snake case.
    """
    if explicit_name:
        return explicit_name
    return _to_snake_case(_strip_suffix(class_name, "Module"))


def _to_snake_case(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _strip_suffix(s: str, optional_suffix: str) -> str:
    if s.endswith(optional_suffix):
        return s[0 : len(s) - len(optional_suffix)]
    else:
        return s
