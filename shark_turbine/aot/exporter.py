# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import overload, Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import io
from pathlib import Path
import platform
import warnings

import torch

from iree.compiler.api import (
    Session,
    Source,
    Output,
)

from ..support.ir_imports import (
    Context,
    Operation,
)

from .builtins import *
from .compiled_module import (
    CompiledModule,
    ImportPhase,
)
from .fx_programs import FxPrograms
from . import decompositions

__all__ = [
    "export",
    "ExportOutput",
]

_is_windows = platform.system() == "Windows"


ModuleLike = Union[
    torch.nn.Module,
    Type[CompiledModule],
    torch.export.ExportedProgram,
    FxPrograms,
]
SaveableTarget = Union[str, Path, None, Output]


class ExportOutput:
    """Wrapper around a CompiledModule produced by `export`."""

    def __init__(
        self,
        session: Session,
        compiled_module: CompiledModule,
        *,
        importer_uses_session: bool = False,
    ):
        self.session = session
        self.session.set_flags("--iree-input-type=torch")
        self.compiled_module = compiled_module
        self._importer_uses_session = importer_uses_session

    @property
    def mlir_module(self) -> Operation:
        """Gets the MLIR module resulting from the last compilation phase."""
        return CompiledModule.get_mlir_module(self.compiled_module)

    def verify(self):
        """Runs the verifier on the module, raising an exception on failure."""
        self.mlir_module.verify()

    def print_readable(self, large_elements_limit: int = 50):
        """Prints a human readable version of the current compilation IR."""
        self.mlir_module.print(large_elements_limit=large_elements_limit)

    def save_mlir(self, file_path: Union[str, Path]):
        """Saves the current compilation IR to a path on disk.

        Args:
            file_path: Path to save the file. If it has a ".mlirbc"
              extension, it will be saved as bytecode. Otherwise as
              text.
        """
        file_path = Path(file_path)
        with open(file_path, "wb") as f:
            if file_path.suffix == ".mlirbc":
                self.mlir_module.write_bytecode(f)
            else:
                self.mlir_module.print(file=f, binary=True)

    def import_to(self, import_to: Union[ImportPhase, str]):
        """Compiles the modules to a mnemonic import phase.

        This is a no-op if already compiled to this phase.
        """
        CompiledModule.run_import(self.compiled_module, import_to)

    def compile(
        self,
        save_to: SaveableTarget,
        *,
        target_backends: Union[str, Sequence[str], None] = ("llvm-cpu",),
    ) -> Optional[memoryview]:
        """Compiles the exported program to an executable binary.

        Args:
            save_to: Where to save the compiled binary. Can be one of:
              None: outputs to a memory buffer and return the API Output.
              (str, Path): Outputs to a file
              Output: Raw compiler API Output object to save to.
            target_backends: A comma-delimitted string of IREE target backends or
              a sequence of strings.
              If `None` does not specify any target backend.
              Then the user must set other appropriate compiler flags e.g.
              `export_output.session.set_flags("--iree-hal-target-device=llvm-cpu")`
        Returns:
          None unless if `save_to=None`, in which case, we return the backing compiler API
          Ouptut object. It can be queried for its backing memory via its `map_memory()`
          method.
        """
        return_memory_view = False
        if save_to is None:
            output = Output.open_membuffer()
            return_memory_view = True
        elif isinstance(save_to, (str, Path)):
            save_to = Path(save_to)
            output = Output.open_file(str(save_to))
        else:
            output = save_to
            assert isinstance(output, Output)

        inv = self.session.invocation()
        if self._importer_uses_session:
            inv.import_module(self.mlir_module)
        else:
            # Some platforms can't share the context across the importer and
            # session (cough: Windows). Round-trip in this case.
            buffer_io = io.BytesIO()
            self.mlir_module.write_bytecode(buffer_io)
            buffer = buffer_io.getvalue()
            source = Source.wrap_buffer(self.session, buffer)
            inv.parse_source(source)
        inv.enable_console_diagnostics()

        # TODO: Don't use flags to set the target backends: set module attributes.
        if target_backends is not None:
            target_backends = (
                target_backends
                if isinstance(target_backends, str)
                else ",".join(target_backends)
            )
            self.session.set_flags(f"--iree-hal-target-backends={target_backends}")
        if not inv.execute():
            raise RuntimeError("Compilation failed: See diagnostics")

        inv.output_vm_bytecode(output)
        output.keep()
        if return_memory_view:
            return output
        else:
            return None


@overload
def export(
    module: torch.nn.Module,
    /,
    *,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Dict[str, Any] | Tuple[Any] | List[Any] | None = None,
    module_name: Optional[str] = None,
    function_name: Optional[str] = None,
    strict_export: bool = True,
) -> ExportOutput:
    """Exports a torch.nn.Module.

    This is done by first invoking torch.export.export with args, kwargs,
    and dynamic_shapes.
    """
    ...


@overload
def export(compiled_module: Type[CompiledModule], /) -> ExportOutput:
    """Exports a CompiledModule and returns the output."""
    ...


@overload
def export(
    exported_program: torch.export.ExportedProgram,
    /,
    *,
    module_name: Optional[str] = None,
    function_name: Optional[str] = None,
) -> ExportOutput:
    """Exports a single entry-point module consisting of an ExportedProgram."""
    ...


@overload
def export(
    exported_programs: FxPrograms,
    /,
    *,
    module_name: Optional[str] = None,
) -> ExportOutput:
    """Exports a multi entry-point ExportedProgram."""
    ...


def export(
    mdl: ModuleLike,
    /,
    *example_args: torch.Tensor,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Dict[str, Any] | Tuple[Any] | List[Any] | None = None,
    module_name: Optional[str] = None,
    function_name: Optional[str] = None,
    strict_export: bool = True,
) -> ExportOutput:
    """Generic export of supported entities.

    See a more specific overload for accepted forms.

    This function behaves differently based on the type of the `mdl` argument:

    * nn.Module: The module is traced with torch.export.export passing it
      `args`, `kwargs`, and `dynamic_shapes`.
    * CompiledModule: The module is imported to IR. Additional arguments are
      illegal in this case.
    * torch.export.ExportedProgram: A pre-exported program can be passed and
      it will be used to construct a single-entrypoint module.

    Args:
      mdl: The nn.Module to export.
      *example_args: Example tensors.
      args: Example arguments to torch.export (if present, then *example_args
        must be empty.
      kwargs: Example keyword arguments.
      dynamic_shapes: Dynamic shape specs to pass to torch.export.

    Returns:
      An ExportOutput object that wraps the compilation and provides
      easy access.
    """
    if len(example_args) > 0:
        warnings.warn(
            DeprecationWarning(
                "extra `example_args` positional parameters are deprecated: pass `args=tuple(...)` instead."
            )
        )

    if not strict_export:
        warnings.warn(
            "When strict_export == False, Python interpreter is used and code will execute exactly as it would in eager mode. "
            "Therefore, graph optimizations and safety are not garaunteed to be at the same level as TorchDynamo. "
            "This is an experimental feature in PyTorch that the IREE Turbine project is still evaluating. Please report issues or experiences."
        )

    TransformedModule: Any
    current_decomps = decompositions.current_aot_decompositions()
    if isinstance(mdl, torch.export.ExportedProgram):
        TransformedModule = CompiledModule.create_from_dict(
            "LambdaCompiledModule",
            {(function_name or "main"): mdl},
            export_name=module_name or "module",
        )

    elif isinstance(mdl, FxPrograms):
        TransformedModule = CompiledModule.create_from_dict(
            "LambdaCompiledModule", mdl.programs, export_name=module_name or "module"
        )
    elif isinstance(mdl, torch.nn.Module):
        # Normalize arguments for torch.export.
        if args is None:
            args = example_args
        elif len(example_args) > 0:
            raise ValueError(
                "Cannot pass args= and positional example_args at the same time"
            )
        nn_module = mdl
        exported_program = torch.export.export(
            nn_module,
            args=args,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict_export,
        )
        if current_decomps:
            from .decompositions import _patch_op_dispatch_for_export

            _patch_op_dispatch_for_export()
            exported_program = exported_program.run_decompositions(current_decomps)

        TransformedModule = CompiledModule.create_from_dict(
            "LambdaCompiledModule",
            {(function_name or "main"): exported_program},
            export_name=module_name or "module",
        )
    elif issubclass(mdl, CompiledModule):
        TransformedModule = mdl
    else:
        raise TypeError(f"mdl argument (type: {type(mdl)}) is not a supported type")

    session = Session()
    # There are some bugs with respect to Session/context interop that we
    # haven't squashed yet. For now, default everyone to round-tripping
    # via bytecode vs sharing the context between the importer/compiler.
    importer_uses_session = False and not _is_windows
    if importer_uses_session:
        context = session.context
    else:
        context = Context()

    cm = TransformedModule(context=context, import_to="import")
    return ExportOutput(session, cm, importer_uses_session=importer_uses_session)
