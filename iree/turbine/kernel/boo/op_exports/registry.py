# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from ..exports.signature import OpSignature
from ..exports.parser import OpCLIParser

from .conv import ConvParser, ConvSignature
from .gemm import GEMMParser, GEMMSignature
from .layer_norm import LayerNormParser, LayerNormSignature


@dataclass
class _BooOpEntry:
    """Bag-of-ops registry entry, value of the registry dictionary."""

    signature_cls: type[OpSignature]
    parser_cls: type[OpCLIParser]


class BooOpRegistry:
    """Bag-of-ops registry mapping a string key of the operation to
    implementation classes and other metadata."""

    _BOO_OP_REGISTRY: dict[str, _BooOpEntry] = {}

    @classmethod
    def get_signature(cls, key: str) -> type[OpSignature] | None:
        """Get the signature class of the op identified by the given key."""
        entry = cls._BOO_OP_REGISTRY.get(key)
        if entry is None:
            return None
        return entry.signature_cls

    @classmethod
    def get_parser(cls, key: str) -> type[OpCLIParser] | None:
        """Get the parser class of the op identified by the given key."""
        entry = cls._BOO_OP_REGISTRY.get(key)
        if entry is None:
            return None
        return entry.parser_cls

    @classmethod
    def parse_command(
        cls, command: str, ignore_unhandled_args: bool = False
    ) -> OpSignature | None:
        """Parse the given command using an op-specific parser selected based on the presence of the op key in the command."""
        key = cls.find_key_from_command(command)
        if key is None:
            return None
        parser_cls = cls.get_parser(key)
        if parser_cls is None:
            return None
        return parser_cls.command_to_signature(
            command, ignore_unhandled_args=ignore_unhandled_args
        )

    @classmethod
    def find_key_from_command(
        cls, command: str, *, assume_unique: bool = True
    ) -> str | None:
        """Find the op key in the given command line.

        If `assume_unique` is set (default), raises an assertion when the
        command contains more than one op key.
        """
        found_key: str | None = None
        for key in cls._BOO_OP_REGISTRY.keys():
            if key not in command:
                continue
            if found_key is not None and assume_unique:
                raise AssertionError(
                    f"Multiple op keys ({found_key}, {key}) are found in the command {command}. This means the command syntax is ambiguous across operations."
                )
            found_key = key
        return found_key

    @classmethod
    def keys(cls) -> list[str]:
        """Get the list of registered op keys."""
        return list(cls._BOO_OP_REGISTRY.keys())

    @classmethod
    def _register(
        cls, key: str, signature_cls: type[OpSignature], parser_cls: type[OpCLIParser]
    ):
        """Register the op."""
        cls._BOO_OP_REGISTRY[key] = _BooOpEntry(signature_cls, parser_cls)


# Register ops.
BooOpRegistry._register("conv", ConvSignature, ConvParser)
BooOpRegistry._register("gemm", GEMMSignature, GEMMParser)
BooOpRegistry._register("layernorm", LayerNormSignature, LayerNormParser)
