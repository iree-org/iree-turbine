# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import abstractmethod, ABC
import functools
from pathlib import Path
import os
import typing
import types

import inspect

from iree.build.executor import ActionConcurrency, BuildAction, BuildContext, BuildFile
from iree.turbine.aot.fx_programs import FxPrograms

__all__ = [
    "turbine_generate",
]


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


class FxProgramsReturnMarshaller(ReturnMarshaller):
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
        if not isinstance(result, FxPrograms):
            raise RuntimeError(
                "Turbine generator was declared to return an FxPrograms instance, "
                f"but it returned {type(result)}"
            )
        import iree.turbine.aot as turbine_aot

        output = turbine_aot.export(result)
        output.save_mlir(path)


RETURN_MARSHALLERS_BY_TYPE = {
    FxPrograms: FxProgramsReturnMarshaller(),
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
        f"In order to wrap a function with @turbine_builder it must be annotated with "
        f"specific return types. Found '{t}' but only {EXPLICIT_MARSHALLER_TYPES} "
        f"are supported"
    )


def unwrap_return_annotation(annot) -> list[ReturnMarshaller]:
    if (
        isinstance(annot, (types.GenericAlias, typing._GenericAlias))
        and annot.__origin__ is tuple
    ):
        unpacked = annot.__args__
    else:
        unpacked = [annot]
    return [get_return_marshaller(it) for it in unpacked]


def turbine_generate(generator: callable, *args, name: str, **kwargs):
    sig = inspect.signature(generator, eval_str=True)
    return_marshallers = unwrap_return_annotation(sig.return_annotation)

    context = BuildContext.current()
    action = TurbineBuilderAction(
        generator,
        args,
        kwargs,
        desc=f"Export turbine model {name}",
        executor=context.executor,
    )
    for rm in return_marshallers:
        rm.prepare_action(context, name, action, len(return_marshallers))
    return [r[1] for r in action.returns]


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
        concurrency=ActionConcurrency.PROCESS,
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
