# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import glob
import importlib
import warnings
from dataclasses import dataclass
from ..exports.signature import OpSignature
from ..exports.parser import OpCLIParser


@dataclass
class _BooOpEntry:
    """Bag-of-ops registry entry, value of the registry dictionary."""

    signature_cls: type[OpSignature]
    parser_cls: type[OpCLIParser]


class BooOpRegistry:
    """Bag-of-ops registry mapping a string key of the operation to
    implementation classes and other metadata."""

    _BOO_OP_REGISTRY: dict[str, _BooOpEntry] = {}

    @staticmethod
    def get_signature(key: str) -> type[OpSignature] | None:
        """Get the signature class of the op identified by the given key."""
        entry = __class__._BOO_OP_REGISTRY.get(key)
        if entry is None:
            return None
        return entry.signature_cls

    @staticmethod
    def get_parser(key: str) -> type[OpCLIParser] | None:
        """Get the parser class of the op identified by the given key."""
        entry = __class__._BOO_OP_REGISTRY.get(key)
        if entry is None:
            return None
        return entry.parser_cls

    @staticmethod
    def parse_command(command: str) -> OpSignature | None:
        """Parse the given command using an op-specific parser selected based on the presence of the op key in the command."""
        key = __class__.find_key_from_command(command)
        if key is None:
            return None
        parser_cls = __class__.get_parser(key)
        return parser_cls.command_to_signature(command)

    @staticmethod
    def find_key_from_command(
        command: str, *, assume_unique: bool = True
    ) -> str | None:
        """Find the op key in the given command line.

        If `assume_unique` is set (default), raises an assertion when the
        command contains more than one op key.
        """
        found_key: str | None = None
        for key in __class__._BOO_OP_REGISTRY.keys():
            if key not in command:
                continue
            if found_key is not None and assume_unique:
                raise AssertionError(
                    f"Multiple op keys ({found_key}, {key}) are found in the command {command}. This means the command syntax is ambiguous across operations."
                )
            found_key = key
        return found_key

    @staticmethod
    def keys() -> list[str]:
        """Get the list of registered op keys."""
        return list(__class__._BOO_OP_REGISTRY.keys())

    @staticmethod
    def _register(
        key: str, signature_cls: type[OpSignature], parser_cls: type[OpCLIParser]
    ):
        """Register the op."""
        __class__._BOO_OP_REGISTRY[key] = _BooOpEntry(signature_cls, parser_cls)


# On module load, traverse all non-private Python files under ../op_exports/,
# import them as modules, and look for the BOO_OP_EXPORT tag to get registration
# info.
__OP_EXPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "op_exports"
)
for file in glob.glob("*.py", root_dir=__OP_EXPORTS_DIR):
    if file.startswith("_"):
        continue
    _module = importlib.import_module(
        "..op_exports." + file[:-3], "iree.turbine.kernel.boo.driver"
    )
    _reg_tuple: tuple[str, type, type] | None = getattr(_module, "BOO_OP_EXPORT", None)
    if _reg_tuple is None:
        warnings.warn(
            "Could not find BOO_OP_EXPORT global variable in "
            + os.path.join(__OP_EXPORTS_DIR, file)
            + ", ignoring."
        )
        continue
    _key, _signature_cls, _parser_cls = _reg_tuple
    BooOpRegistry._register(_key, _signature_cls, _parser_cls)
