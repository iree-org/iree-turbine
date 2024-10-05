# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helpers for implementing ops.

Typical usage:

```
  _templates = JinjaTemplateLoader(__name__)

  def generate(kb: KernelBuilder):
    func_op = _templates.inline_template_function(
        kb, "my_template", "function_name", **kwargs)
    return call_function(func_op, *values)
```
"""

from typing import Sequence

from abc import ABC, abstractmethod
import logging
import textwrap

from ...support.logging import runtime_logger as logger

from ...support.ir_imports import (
    FlatSymbolRefAttr,
    FunctionType,
    MLIRError,
    Operation,
    StringAttr,
    TypeAttr,
    Value,
)

from ...transforms.merger import Merger

from .base import (
    KernelBuilder,
)


__all__ = [
    "TemplateLoader",
    "StrFormatTemplateLoader",
    "call_function",
]


class TemplateLoader(ABC):
    """Base class for templates that can be loaded by name."""

    @abstractmethod
    def load_template(self, kb: KernelBuilder, name: str, **kwargs) -> Operation:
        """Loads a template by name and kwargs, returning the module operation."""
        ...

    def _parse_module_asm(self, kb: KernelBuilder, asm: str) -> Operation:
        try:
            module_op = Operation.parse(asm, context=kb.context)
        except MLIRError as e:
            lines = asm.splitlines()
            lines_numbered = "\n".join(
                [f"      {str(i+1):>5}: {l}" for i, l in enumerate(lines)]
            )
            raise RuntimeError(
                f"Error parsing generated op template:"
                f"\n{textwrap.indent(str(e), '  ')}"
                f"\n{lines_numbered}"
            )
        return module_op.operation

    def inline_template_function(
        self,
        kb: KernelBuilder,
        template_file: str,
        function_name: str,
        **kwargs,
    ) -> Operation:
        """Inlines a template module by first expanding its ASM via **kwargs.

        Returns the inlined symbol `function_name`, which is expected to have been
        in the template.
        """
        try:
            return kb.symbol_table[function_name]
        except KeyError:
            pass
        source_module_op = self.load_template(kb, template_file, **kwargs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Generated kernel IR %s:\n%s", function_name, str(source_module_op)
            )
        merger = Merger(
            source_module_op, kb.module_body.owner, target_symbol_table=kb.symbol_table
        )
        merger.merge()
        return kb.symbol_table[function_name]


class StrFormatTemplateLoader(TemplateLoader):
    """Template loader that uses str.format.

    Usage:
      _templates = StrFromatTemplateLoader(__name__)

    By default, this will resolve a template like "foo" from templates/foo.mlir
    in the package directory.
    """

    def __init__(
        self,
        package_name: str,
        package_path: str = "templates",
        *,
        suffix: str = ".mlir",
    ):
        self.parent_package_name = ".".join(package_name.split(".")[0:-1])
        self.package_path = package_path
        self.suffix = suffix

    def load_template(self, kb: KernelBuilder, name: str, **kwargs) -> Operation:
        from importlib import resources

        res = (
            resources.files(self.parent_package_name)
            / self.package_path
            / f"{name}{self.suffix}"
        )
        contents = res.read_text().format(**kwargs)
        return self._parse_module_asm(kb, contents)


class JinjaTemplateLoader(TemplateLoader):
    """Template loader based on jinja templates.

    Usage:
      _templates = JinjaTemplateLoader(__name__)

    By default, this will resolve a template like "foo" from templates/foo.mlir
    in the package directory.
    """

    def __init__(
        self,
        package_name: str,
        package_path: str = "templates",
        *,
        suffix: str = ".mlir",
    ):
        try:
            from jinja2 import Environment, PackageLoader, select_autoescape
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Cannot use JinjaTemplateLoader if jinja2 is not installed"
            ) from e
        self.env = Environment(loader=PackageLoader(package_name, package_path))
        self.suffix = suffix

    def load_template(self, kb: KernelBuilder, name: str, **kwargs) -> Operation:
        template_file = f"{name}{self.suffix}"
        contents = self.env.get_template(template_file).render(**kwargs)
        return self._parse_module_asm(kb, contents)


def call_function(target_function: Operation, *operands: Value) -> Sequence[Value]:
    """Emits a util.call for a util.func target function operation."""
    target_symbol = FlatSymbolRefAttr.get(
        StringAttr(target_function.attributes["sym_name"]).value_bytes
    )
    ftype = FunctionType(TypeAttr(target_function.attributes["function_type"]).value)
    return Operation.create(
        "util.call",
        results=ftype.results,
        operands=operands,
        attributes={
            "callee": target_symbol,
        },
    ).results
