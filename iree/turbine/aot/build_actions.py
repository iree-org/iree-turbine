# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable

from abc import abstractmethod, ABC
from pathlib import Path
import typing
import types

import inspect

from iree.build.executor import ActionConcurrency, BuildAction, BuildContext, BuildFile
from iree.turbine.aot.exporter import ExportOutput

__all__ = [
    "turbine_generate",
]


def turbine_generate(
    generator: Callable,
    *args,
    name: str,
    out_of_process: bool = True,
    **kwargs,
):
    """Invokes a user-defined generator callable as an action, performing turbine
    import and storing the resulting artifacts as outputs.

    Because torch-based generation is usually quite slow and a bottleneck, this
    action takes pains to use the out of process action pool, allowing multiple
    generation activities to take place concurrently. Since this requires interacting
    with the pickle infrastructure, it puts some constraints on usage:

    * generator must be a pickleable callable. In practice, this means that it must
      be a named function at module scope (without decorator) or a named class at
      module scope with a `__call__` method.
    * args and kwargs must be pickleable. In practice, this means primitive values.

    Arguments to the generator are taken from the positional and unmatched keyword
    arguments passed to `turbine_generate`.

    The generator makes artifacts available as outputs by returning corresponding
    Python instances (which must be declared as typing parameters for the remoting
    to work):

    * `ExportOutput`: The result of calling `aot.export(...)` will result in
      `save_mlir()` being called on it while still in the subprocess to write to
      a file names `{name}.mlir` if there is one return or `{name}_{n}.mlir` if
      multiple.

    By default, import is run in a subprocess pool. It can be run in the main
    process by passing `out_of_process=False`.

    See testing/example_builder.py for an example.
    """
    sig = inspect.signature(generator, eval_str=True)
    return_marshallers = unwrap_return_annotation(sig.return_annotation)

    context = BuildContext.current()
    action = TurbineBuilderAction(
        generator,
        args,
        kwargs,
        desc=f"Export turbine model {name}",
        executor=context.executor,
        concurrency=(
            ActionConcurrency.PROCESS if out_of_process else ActionConcurrency.THREAD
        ),
    )
    for rm in return_marshallers:
        rm.prepare_action(context, name, action, len(return_marshallers))
    return [r[1] for r in action.returns]


class ReturnMarshaller(ABC):
    @abstractmethod
    def prepare_action(
        self,
        context: BuildContext,
        name: str,
        action: "TurbineBuilderAction",
        return_arity: int,
    ):
        ...

    @abstractmethod
    def save_remote_result(self, result, path: Path):
        ...


class ExportOutputReturnMarshaller(ReturnMarshaller):
    def prepare_action(
        self,
        context: BuildContext,
        name: str,
        action: "TurbineBuilderAction",
        return_arity: int,
    ):
        # Need to allocate one file for output.
        file_name = (
            f"{name}_{len(action.returns)}.mlir" if return_arity > 1 else f"{name}.mlir"
        )
        output_file = context.allocate_file(file_name)
        action.returns.append((self, output_file))
        output_file.deps.add(action)

    def save_remote_result(self, result, path: Path):
        if not isinstance(result, ExportOutput):
            raise RuntimeError(
                "Turbine generator was declared to return an ExportOutput instance, "
                f"but it returned {type(result)}"
            )
        result.save_mlir(path)


RETURN_MARSHALLERS_BY_TYPE: dict[type, ReturnMarshaller] = {
    ExportOutput: ExportOutputReturnMarshaller(),
}
EXPLICIT_MARSHALLER_TYPES = list(RETURN_MARSHALLERS_BY_TYPE.keys())


def get_return_marshaller(t: type) -> ReturnMarshaller:
    m = RETURN_MARSHALLERS_BY_TYPE.get(t)
    if m is not None:
        return m

    # Do an exhaustive subclass check.
    for k, m in RETURN_MARSHALLERS_BY_TYPE.items():
        if issubclass(t, k):
            # Cache it.
            RETURN_MARSHALLERS_BY_TYPE[t] = m
            return m
    raise ValueError(
        f"In order to use a callable as a generator in turbine_generate, it must be "
        f" annotated with specific return types. Found '{t}' but only "
        f"{EXPLICIT_MARSHALLER_TYPES} are supported"
    )


def unwrap_return_annotation(annot) -> list[ReturnMarshaller]:
    # typing._GenericAlias is used to unwrap old-style (i.e. `List`) collection
    # aliases. We special case this and it can be removed eventually.
    _GenericAlias = getattr(typing, "_GenericAlias", None)
    is_generic_alias = isinstance(annot, (types.GenericAlias)) or (
        _GenericAlias and isinstance(annot, _GenericAlias)
    )
    if is_generic_alias and annot.__origin__ is tuple:
        unpacked = annot.__args__
    else:
        unpacked = [annot]
    return [get_return_marshaller(it) for it in unpacked]


class RemoteGenerator:
    def __init__(
        self,
        generation_thunk,
        thunk_args,
        thunk_kwargs,
        return_info: list[tuple[ReturnMarshaller, Path]],
    ):
        self.generation_thunk = generation_thunk
        self.thunk_args = thunk_args
        self.thunk_kwargs = thunk_kwargs
        self.return_info = return_info

    def __call__(self):
        results = self.generation_thunk(*self.thunk_args, **self.thunk_kwargs)
        if not isinstance(results, (tuple, list)):
            results = [results]
        if len(results) != len(self.return_info):
            raise RuntimeError(
                f"Turbine generator {self.generation_thunk} returned {len(results)} values, "
                f"but it was declared to return {len(self.return_info)}"
            )
        for result, (marshaller, output_path) in zip(results, self.return_info):
            marshaller.save_remote_result(result, output_path)


class TurbineBuilderAction(BuildAction):
    def __init__(
        self,
        thunk,
        thunk_args,
        thunk_kwargs,
        concurrency,
        **kwargs,
    ):
        super().__init__(concurrency=concurrency, **kwargs)
        self.thunk = thunk
        self.thunk_args = thunk_args
        self.thunk_kwargs = thunk_kwargs
        self.returns: list[tuple[ReturnMarshaller, BuildFile]] = []

    def _remotable_thunk(self):
        remotable_return_info = [
            (marshaller, bf.get_fs_path()) for marshaller, bf in self.returns
        ]
        return RemoteGenerator(
            self.thunk, self.thunk_args, self.thunk_kwargs, remotable_return_info
        )
