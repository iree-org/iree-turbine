# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import io
from pathlib import Path
import platform

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
    CompiledModuleMeta,
    ImportPhase,
)


_is_windows = platform.system() == "Windows"


ModuleLike = Union[torch.nn.Module, CompiledModuleMeta, torch.export.ExportedProgram]
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
        target_backends: Union[str, Sequence[str]] = ("llvm-cpu",),
    ) -> Optional[memoryview]:
        """Compiles the exported program to an executable binary.

        Args:
            save_to: Where to save the compiled binary. Can be one of:
              None: outputs to a memory buffer and return the API Output.
              (str, Path): Outputs to a file
              Output: Raw compiler API Output object to save to.
            target_backends: A comma-delimitted string of IREE target backends or
              a sequence of strings.
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

        target_backends = (
            target_backends
            if isinstance(target_backends, str)
            else ",".join(target_backends)
        )
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
        self.session.set_flags(f"--iree-hal-target-backends={target_backends}")
        if not inv.execute():
            raise RuntimeError("Compilation failed: See diagnostics")

        inv.output_vm_bytecode(output)
        output.keep()
        if return_memory_view:
            return output
        else:
            return None


def export(
    mdl: ModuleLike,
    *example_args: torch.Tensor,
    args: Optional[tuple] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Dict[str, Any] | Tuple[Any] | List[Any] | None = None,
    external_params: bool = False,
) -> ExportOutput:
    """One shot export of an nn.Module or CompiledModule.

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
      external_params: Whether to declare parameters as external vs inlining
        contents.

    Returns:
      An ExportOutput object that wraps the compilation and provides
      easy access.
    """
    TransformedModule: Any
    if isinstance(mdl, torch.export.ExportedProgram):
        if (
            len(example_args) > 0
            or args is not None
            or kwargs is not None
            or dynamic_shapes is not None
        ):
            raise ValueError(
                "If passing an ExportedProgram to aot.export, cannot also pass "
                "args, example_args, kwargs, or dynamic_dims"
            )

        class EpExported(CompiledModule, export_name=mdl.graph_module._get_name()):
            params = export_global_tree(
                dict(list(mdl.named_parameters())), external=external_params
            )
            main = mdl

        TransformedModule = EpExported
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
            nn_module, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
        )

        class Exported(CompiledModule, export_name=nn_module._get_name()):
            params = export_parameters(nn_module, external=external_params)
            main = exported_program

        TransformedModule = Exported
    else:
        assert isinstance(mdl, CompiledModuleMeta)
        if (
            len(example_args) > 0
            or args is not None
            or kwargs is not None
            or dynamic_shapes is not None
        ):
            raise ValueError(
                "If passing a CompiledModule to aot.export, cannot also pass "
                "args, example_args, kwargs, or dynamic_dims"
            )
        TransformedModule = mdl

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
