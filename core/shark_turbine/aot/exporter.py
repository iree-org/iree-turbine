# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence, Union
import functools
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
    ExportProcDef,
    ImportPhase,
)
from .support.procedural import (
    AbstractTypedef,
)


_is_windows = platform.system() == "Windows"


ModuleLike = Union[torch.nn.Module, CompiledModuleMeta]
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
            assert isinstance(output, Output)
            output = save_to

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


# Decorator which explicitly exports a function.
# TODO: Make this a public API on CompiledModule.
# See https://github.com/nod-ai/SHARK-Turbine/issues/126
def export_proc(f=None, *, signature: Sequence[AbstractTypedef]) -> ExportProcDef:
    if f is None:
        return functools.partial(export_proc, signature=signature)
    return ExportProcDef(f.__name__, f, signature=signature)


def export(mdl: ModuleLike, *example_args: torch.Tensor) -> ExportOutput:
    """One shot export of an nn.Module.

    This is a very restrictive API vs the lower level `CompiledModule`
    facility. It is suitable for one-shot modules, with a single
    entrypoint and static example arguments where no additional
    configuration is needed for mutable parameters/buffers or state
    management. Dynamic shape constraints are also not presently
    exposed via this API, but we expect to allow this in the future.

    Args:
      mdl: The nn.Module to export.
      *example_args: Example tensors.

    Returns:
      An ExportOutput object that wraps the compilation and provides
      easy access.
    """
    if isinstance(mdl, torch.nn.Module):
        signature = [abstractify(t) for t in example_args]

        class Exported(CompiledModule, export_name=mdl._get_name()):
            params = export_parameters(mdl)

            @export_proc(signature=signature)
            def main(self, *args):
                return jittable(mdl.forward)(*args)

    else:
        assert isinstance(mdl, CompiledModuleMeta)
        Exported = mdl

    session = Session()
    # There are some bugs with respect to Session/context interop that we
    # haven't squashed yet. For now, default everyone to round-tripping
    # via bytecode vs sharing the context between the importer/compiler.
    importer_uses_session = False and not _is_windows
    if importer_uses_session:
        context = session.context
    else:
        context = Context()

    cm = Exported(context=context, import_to="import")
    return ExportOutput(session, cm, importer_uses_session=importer_uses_session)
